# @testset "Disk cache" begin
#     @test GPUCompiler.disk_cache() == false
#     cmd = Base.julia_cmd()
#     if Base.JLOptions().project != C_NULL
#         cmd = `$cmd --project=$(unsafe_string(Base.JLOptions().project))`
#     end

#     withenv("JULIA_LOAD_PATH" => "$(get(ENV, "JULIA_LOAD_PATH", "")):$(joinpath(@__DIR__, "CacheEnv"))") do
#         @test success(pipeline(`$cmd cache.jl true`, stderr=stderr, stdout=stdout))
#         @test success(pipeline(`$cmd cache.jl false`, stderr=stderr, stdout=stdout))
#     end
# end


using GPUCompiler
using Test

const TOTAL_KERNELS = 1

clear = parse(Bool, ARGS[1])

@test GPUCompiler.disk_cache() == true

if clear
    GPUCompiler.clear_disk_cache!()
    @test length(readdir(GPUCompiler.cache_path())) == 0
else
    @test length(readdir(GPUCompiler.cache_path())) == TOTAL_KERNELS
end

using LLVM, LLVM.Interop

include("util.jl")
include("definitions/native.jl")

kernel() = return

const runtime_cache = Dict{UInt, Any}()

function compiler(job)
    return GPUCompiler.compile(:asm, job)
end

function linker(job, asm)
    asm
end

let (job, kwargs) = native_job(kernel, Tuple{})
    source = job.source
    config = job.config
    GPUCompiler.cached_compilation(runtime_cache, config, source.ft, source.tt, compiler, linker)
end

@test length(readdir(GPUCompiler.cache_path())) == TOTAL_KERNELS
