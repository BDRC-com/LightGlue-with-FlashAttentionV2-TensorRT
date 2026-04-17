#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>

#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

    class Logger : public nvinfer1::ILogger
    {
    public:
        void log(Severity severity, const char *msg) noexcept override
        {
            if (severity > Severity::kINFO)
            {
                return;
            }
            std::cerr << "[TensorRT] " << msg << std::endl;
        }
    };

    template <typename T>
    struct Destroy
    {
        void operator()(T *obj) const
        {
            delete obj;
        }
    };

    template <typename T>
    using TrtUniquePtr = std::unique_ptr<T, Destroy<T>>;

    struct Options
    {
        std::string onnx_path;
        std::string engine_path;
        bool enable_fp16 = false;
        bool enable_best = false;
        size_t workspace_mib = 1024;
    };

    void ensure(bool condition, const std::string &message)
    {
        if (!condition)
        {
            throw std::runtime_error(message);
        }
    }

    size_t parseSizeT(const char *value, const std::string &flag_name)
    {
        try
        {
            return static_cast<size_t>(std::stoull(value));
        }
        catch (const std::exception &)
        {
            throw std::runtime_error("invalid value for " + flag_name + ": " + value);
        }
    }

    Options parseArgs(int argc, char **argv)
    {
        Options options;
        for (int i = 1; i < argc; ++i)
        {
            const std::string arg = argv[i];
            if (arg == "--onnx" && i + 1 < argc)
            {
                options.onnx_path = argv[++i];
            }
            else if (arg == "--engine" && i + 1 < argc)
            {
                options.engine_path = argv[++i];
            }
            else if (arg == "--fp16")
            {
                options.enable_fp16 = true;
            }
            else if (arg == "--best")
            {
                options.enable_best = true;
                options.enable_fp16 = true;
            }
            else if (arg == "--workspace-mib" && i + 1 < argc)
            {
                options.workspace_mib = parseSizeT(argv[++i], "--workspace-mib");
            }
            else if (arg == "-h" || arg == "--help")
            {
                std::cout
                    << "Usage: build_trt_engine_from_onnx --onnx PATH --engine PATH [--fp16|--best] [--workspace-mib N]\n";
                std::exit(0);
            }
            else
            {
                throw std::runtime_error("unknown argument: " + arg);
            }
        }

        ensure(!options.onnx_path.empty(), "missing required --onnx");
        ensure(!options.engine_path.empty(), "missing required --engine");
        return options;
    }

    std::string dimsToString(const nvinfer1::Dims &dims)
    {
        std::ostringstream oss;
        for (int i = 0; i < dims.nbDims; ++i)
        {
            if (i > 0)
            {
                oss << 'x';
            }
            oss << dims.d[i];
        }
        return oss.str();
    }

    void printNetworkSummary(nvinfer1::INetworkDefinition &network)
    {
        std::cout << "Network inputs:" << std::endl;
        for (int i = 0; i < network.getNbInputs(); ++i)
        {
            const auto *tensor = network.getInput(i);
            std::cout << "  - " << tensor->getName() << " : "
                      << dimsToString(tensor->getDimensions()) << std::endl;
        }

        std::cout << "Network outputs:" << std::endl;
        for (int i = 0; i < network.getNbOutputs(); ++i)
        {
            const auto *tensor = network.getOutput(i);
            std::cout << "  - " << tensor->getName() << " : "
                      << dimsToString(tensor->getDimensions()) << std::endl;
        }
    }

} // namespace

int main(int argc, char **argv)
{
    try
    {
        const Options options = parseArgs(argc, argv);

        std::ifstream onnx_file(options.onnx_path, std::ios::binary);
        ensure(onnx_file.good(), "failed to open ONNX: " + options.onnx_path);

        Logger logger;
        initLibNvInferPlugins(&logger, "");

        TrtUniquePtr<nvinfer1::IBuilder> builder(
            nvinfer1::createInferBuilder(logger));
        ensure(builder != nullptr, "failed to create TensorRT builder");

        const auto network_flags = 1U << static_cast<uint32_t>(
                                       nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        TrtUniquePtr<nvinfer1::INetworkDefinition> network(
            builder->createNetworkV2(network_flags));
        ensure(network != nullptr, "failed to create TensorRT network");

        TrtUniquePtr<nvonnxparser::IParser> parser(
            nvonnxparser::createParser(*network, logger));
        ensure(parser != nullptr, "failed to create ONNX parser");

        const bool parsed = parser->parseFromFile(
            options.onnx_path.c_str(),
            static_cast<int>(nvinfer1::ILogger::Severity::kINFO));
        ensure(parsed, "failed to parse ONNX: " + options.onnx_path);

        printNetworkSummary(*network);

        TrtUniquePtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
        ensure(config != nullptr, "failed to create TensorRT builder config");

        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                                   options.workspace_mib * (1ULL << 20));

        if (options.enable_fp16 && builder->platformHasFastFp16())
        {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
            std::cout << "FP16 enabled" << std::endl;
        }
        else if (options.enable_fp16)
        {
            std::cout << "FP16 requested but platformHasFastFp16() is false; continuing with FP32"
                      << std::endl;
        }

        if (options.enable_best)
        {
            config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
        }

        TrtUniquePtr<nvinfer1::IHostMemory> serialized_engine(
            builder->buildSerializedNetwork(*network, *config));
        ensure(serialized_engine != nullptr, "failed to build serialized TensorRT engine");

        std::ofstream engine_file(options.engine_path, std::ios::binary);
        ensure(engine_file.good(), "failed to open output engine path: " + options.engine_path);
        engine_file.write(static_cast<const char *>(serialized_engine->data()),
                          static_cast<std::streamsize>(serialized_engine->size()));
        engine_file.close();

        std::cout << "Wrote engine: " << options.engine_path << std::endl;
        std::cout << "Engine bytes: " << serialized_engine->size() << std::endl;
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}