class TRTLogger : public nvinfer1::ILogger{
    public:
        void log(Severity severity, const char* msg) noexcept override;
};

inline TRTLogger gLogger;

