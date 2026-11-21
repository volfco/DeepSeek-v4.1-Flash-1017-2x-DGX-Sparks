"""Adapt the installed NVIDIA PM-sampling example to record device-wide counters without launching kernels."""
import pathlib
import shutil
import sys

source = pathlib.Path("/usr/local/cuda/extras/CUPTI/samples/pm_sampling")
dest = pathlib.Path(sys.argv[1])
dest.mkdir(exist_ok=True)
shutil.copyfile(source/"Makefile", dest/"Makefile")
cpp = (source/"pm_sampling.cu").read_text()
cpp = "#include <cupti_activity.h>\n" + cpp
start=cpp.index("    const size_t NUM_OF_ITERATIONS")
end=cpp.index("    // 5. Stop", start)
cpp=cpp[:start]+'''    uint64_t gpu_now = 0;
    CUPTI_API_CALL(cuptiGetTimestamp(&gpu_now));
    auto utc = std::chrono::system_clock::now().time_since_epoch();
    std::cout << "anchor," << gpu_now << "," << std::chrono::duration_cast<std::chrono::nanoseconds>(utc).count() << std::endl;
    const char *seconds = std::getenv("COUNTER_SECONDS");
    std::this_thread::sleep_for(std::chrono::seconds(seconds ? std::min(900, std::max(1, std::atoi(seconds))) : 5));
'''+cpp[end:]
cpp=cpp.replace("    vectorWorkLoad.SetUp();","").replace("    vectorWorkLoad.TearDown();","")
cpp=cpp.replace("    while (!stopDecodeThread)\n    {","    while (!stopDecodeThread)\n    {\n        std::this_thread::sleep_for(std::chrono::milliseconds(50));")
cpp=cpp.replace("CUptiResult threadFuncResult;","CUptiResult threadFuncResult = CUPTI_SUCCESS;")
(dest/"pm_sampling.cu").write_text(cpp)
hdr=(source/"pm_sampling.h").read_text()
start=hdr.index('        std::cout << "Total num of Samples')
end=hdr.index("    CUptiResult GetSupportedBaseMetrics",start)
hdr=hdr[:start]+'''        for (const auto& range : m_samplerRanges) {
            for (const auto& metric : range.metricValues) {
                std::cout << range.startTimestamp << "," << range.endTimestamp << "," << metric.first << "," << std::setprecision(15) << metric.second << "\\n";
            }
        }
    }

'''+hdr[end:]
(dest/"pm_sampling.h").write_text(hdr)
