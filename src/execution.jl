# reusable functionality to implement code execution

export split_kwargs, assign_args!


## macro tools

# split keyword arguments expressions into groups. returns vectors of keyword argument
# values, one more than the number of groups (unmatched keywords in the last vector).
# intended for use in macros; the resulting groups can be used in expressions.
function split_kwargs(kwargs, kw_groups...)
    kwarg_groups = ntuple(_->[], length(kw_groups) + 1)
    for kwarg in kwargs
        # decode
        Meta.isexpr(kwarg, :(=)) || throw(ArgumentError("non-keyword argument like option '$kwarg'"))
        key, val = kwarg.args
        isa(key, Symbol) || throw(ArgumentError("non-symbolic keyword '$key'"))

        # find a matching group
        group = length(kwarg_groups)
        for (i, kws) in enumerate(kw_groups)
            if key in kws
                group = i
                break
            end
        end
        push!(kwarg_groups[group], kwarg)
    end

    return kwarg_groups
end

# assign arguments to variables, handle splatting
function assign_args!(code, _args)
    nargs = length(_args)

    # handle splatting
    splats = Vector{Bool}(undef, nargs)
    args = Vector{Any}(undef, nargs)
    for i in 1:nargs
        splats[i] = Meta.isexpr(_args[i], :(...))
        args[i] = splats[i] ? _args[i].args[1] : _args[i]
    end

    # assign arguments to variables
    vars = Vector{Symbol}(undef, nargs)
    for i in 1:nargs
        vars[i] = gensym()
        push!(code.args, :($(vars[i]) = $(args[i])))
    end

    # convert the arguments, compile the function and call the kernel
    # while keeping the original arguments alive
    var_exprs = Vector{Any}(undef, nargs)
    for i in 1:nargs
        var_exprs[i] = splats[i] ? Expr(:(...), vars[i]) : vars[i]
    end

    return vars, var_exprs
end


## cached compilation
disk_cache() = parse(Bool, @load_preference("disk_cache", "false"))

"""
    enable_cache!(state::Bool=true)

Activate the GPUCompiler disk cache in the current environment.
You will need to restart your Julia environment for it to take effect.

!!! note
    The cache functionality requires Julia 1.11
"""
function enable_cache!(state::Bool=true)
    @set_preferences!("disk_cache"=>string(state))
end

cache_path() = @get_scratch!("cache")
clear_disk_cache!() = rm(cache_path(); recursive=true, force=true)
function cache_path(key)
    return joinpath(
        cache_path(),
        # TODO: Use object_build_id from https://github.com/JuliaLang/julia/pull/53943
        #       Should we disk cache "runtime compilation".
        string(Base.module_build_id(GPUCompiler)), # captures dependencies as well
        string(cache_key), "ir.jls")
end

const cache_lock = ReentrantLock()

"""
    cached_compilation(cache::Dict{Any}, src::MethodInstance, cfg::CompilerConfig,
                       compiler, linker)

Compile a method instance `src` with configuration `cfg`, by invoking `compiler` and
`linker` and storing the result in `cache`.

The `cache` argument should be a dictionary that can be indexed using any value and store
whatever the `linker` function returns. The `compiler` function should take a `CompilerJob`
and return data that can be cached across sessions (e.g., LLVM IR). This data is then
forwarded, along with the `CompilerJob`, to the `linker` function which is allowed to create
session-dependent objects (e.g., a `CuModule`).
"""
function cached_compilation(cache::AbstractDict{<:Any,V},
                            src::MethodInstance, cfg::CompilerConfig,
                            compiler::Function, linker::Function) where {V}
    # NOTE: we index the cach both using (mi, world, cfg) keys, for the fast look-up,
    #       and using CodeInfo keys for the slow look-up. we need to cache both for
    #       performance, but cannot use a separate private cache for the ci->obj lookup
    #       (e.g. putting it next to the CodeInfo's in the CodeCache) because some clients
    #       expect to be able to wipe the cache (e.g. CUDA.jl's `device_reset!`)

    # fast path: index the cache directly for the *current* world + compiler config

    world = tls_world_age()
    key = (objectid(src), world, cfg)
    # NOTE: we store the MethodInstance's objectid to avoid an expensive allocation.
    #       Base does this with a multi-level lookup, first keyed on the mi,
    #       then a linear scan over the (typically few) entries.

    # NOTE: no use of lock(::Function)/@lock/get! to avoid try/catch and closure overhead
    lock(cache_lock)
    obj = get(cache, key, nothing)
    unlock(cache_lock)

    if obj === nothing || compile_hook[] !== nothing
        obj = actual_compilation(cache, src, world, cfg, compiler, linker)::V
        lock(cache_lock)
        cache[key] = obj
        unlock(cache_lock)
    end
    return obj::V
end

@noinline function actual_compilation(cache::AbstractDict, src::MethodInstance, world::UInt,
                                      cfg::CompilerConfig, compiler::Function, linker::Function)
    job = CompilerJob(src, cfg, world)
    obj = nothing

    # fast path: find an applicable CodeInstance and see if we have compiled it before
    ci = ci_cache_lookup(ci_cache(job), src, world, world)::Union{Nothing,CodeInstance}
    if ci !== nothing
        obj = get(cache, ci, nothing)
    end

    # slow path: compile and link
    if obj === nothing || compile_hook[] !== nothing
        if obj !== nothing
            # we got here because of a *compile* hook; don't bother linking
            return obj
        end
        asm = nothing
        @static if VERSION >= v"1.11.0-" && disk_cache()
            cache_key = Base.objectid(ci)
            path = cache_path(cache_key)
            if isfile(path)
                try
                    @debug "Loading compiled kernel for $spec from $path"
                    asm = deserialize(path)
                catch ex
                    @warn "Failed to load compiled kernel at $path" exception=(ex, catch_backtrace())
                end
            end
        else

        if asm === nothing
            asm = compiler(job)
        end

        @static if VERSION >= v"1.11.0-" && disk_cache() && !isfile(path)
            # TODO: Should we only write out during precompilation?
            tmppath, io = mktemp(;cleanup=false)
            serialize(io, asm)
            close(io)
            # atomic move
            Base.rename(tmppath, path, force=true)
        end

        obj = linker(job, asm)
        ci = ci_cache_lookup(ci_cache(job), src, world, world)::CodeInstance
        cache[ci] = obj
    end

    return obj
end
