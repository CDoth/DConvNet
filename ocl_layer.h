#ifndef OCL_LAYER_H
#define OCL_LAYER_H
#include <DSynapse.h>
#include <net.h>
#include <DArray.h>

#include <CL/cl.h>
#include <CL/cl.hpp>
using namespace DSynapse;

typedef cl::Buffer DEVICEBUFFER, *PDEVICEBUFFER;
typedef float target_type;
struct OCLContext
{
    std::vector<cl::Device> all_devices;
    cl::Device device;
    cl::Context context;
    cl::Program program;
};
struct cl_ConvolutionContext
{
    int w;
    int h;
    int channels_in;
    int channels_out;
    int kw;
    int kh;

    int padding_w;
    int padding_h;
    int step_w;
    int step_h;
    int stride_w;
    int stride_h;

    /// \brief activation_type
    /// 0: Sigmoid
    /// 1: gtan
    /// 2: ReLU
    ACTIVATION activation_type;

    bool use_common_kernel;
    bool use_act_signal;

    target_type learn_rate;

    void (*set_rand)(target_type& v);
};
struct cl_PoolingContext
{
    int w;
    int h;
    int pw;
    int ph;
    int pooling_type; //0 - max, 1 - mean

    int channels;
    void (*set_rand)(target_type& v);
};

enum LayerBufferType{
     BUFFERTYPE_INPUT
    ,BUFFERTYPE_RAW_INPUT
    ,BUFFERTYPE_INPUT_ERROR
    ,BUFFERTYPE_DERACT_OUT
    ,BUFFERTYPE_OUTPUT
    ,BUFFERTYPE_RAW_OUTPUT
    ,BUFFERTYPE_OUTPUT_ERROR
    ,BUFFERTYPE_KERNELS
    ,BUFFERTYPE_DELTA
    ,BUFFERTYPE_BIAS
    ,BUFFERTYPE_BIAS_DELTA
};
typedef struct HOSTBUFFER
{
    HOSTBUFFER() : loaded(0) {}
    bool epmty() const {return channels.size() == 0;}
    DMultiMatrix<target_type> channels;
    int loaded;
} *PHOSTBUFFER;

/*
class BaseLayer
{
public:
    BaseLayer();
    virtual void forward_propagation() = 0;
    virtual void alloc_output() = 0;
    virtual void build_up() = 0;
    virtual void setRandInput() = 0;
    virtual void setInput(const DMultiMatrix<target_type> &data) = 0;
    virtual void setInput(const target_type *data, int size = 0) = 0;

    virtual void loadHostBuffer(LayerHostBuffers buf) = 0;
    virtual DMultiMatrix<target_type> getAllChannels(LayerHostBuffers t) = 0;
    virtual DMatrix<target_type> getChannel(LayerHostBuffers t, int ch) = 0;
    virtual void showAllChannels(LayerHostBuffers t, const char *message = nullptr) = 0;

    int getDeviceMem() const;
    const char *getDeviceMemShortSize();
    int getHostMem() const;
    const char *getHostMemShortSize();

protected:
    virtual PHOSTBUFFER _getHostBuffer(LayerHostBuffers t) = 0;
    virtual PDEVICEBUFFER _getDeviceBuffer(LayerHostBuffers t) = 0;
    virtual int _getHostChannelsNumber(LayerHostBuffers t) = 0;


    void allocDeviceBuffer(DEVICEBUFFER &buffer, int size);
    void allocDeviceBuffer(PDEVICEBUFFER &buffer, int size);
    void allocHostBuffer(HOSTBUFFER &buffer, int channels_n, int w, int h);
    void allocHostBuffer(PHOSTBUFFER &buffer, int channels_n, int w, int h);
    virtual void allocHostBuffer(LayerHostBuffers t) = 0;

    void copyToDeviceBuffer(DEVICEBUFFER &DevBuffer, const target_type *HostBuffer, int size);
    void copyToDeviceBuffer(PDEVICEBUFFER &DevBuffer, const target_type *HostBuffer, int size);
    void copyToHostBuffer(DEVICEBUFFER &DevBuffer, target_type *HostBuffer, int size);
    void copyToHostBuffer(PDEVICEBUFFER &DevBuffer, target_type *HostBuffer, int size);

    void copyToDeviceBuffer(DEVICEBUFFER &DevBuffer, PHOSTBUFFER HostBuffer);
    void copyToDeviceBuffer(PDEVICEBUFFER &DevBuffer, PHOSTBUFFER HostBuffer);
    void copyToHostBuffer(DEVICEBUFFER &DevBuffer, PHOSTBUFFER HostBuffer);
    void copyToHostBuffer(PDEVICEBUFFER &DevBuffer, PHOSTBUFFER HostBuffer);

    OCLContext *ocl;
    cl::CommandQueue MainQueue;
    int DeviceMem;
    char deviceMemShortSize[20];
    int HostMem;
    char hostMemShortSize[20];


    int w_out;
    int h_out;
};
*/

struct _back_conv_param
{
    int KernelStartPoint;
    int OutputStartPoint;
    int n;
};
struct _back_conv_param2
{
    int KernelStartPoint_w;
    int OutputStartPoint_w;
    int n_w;

    int KernelStartPoint_h;
    int OutputStartPoint_h;
    int n_h;
};
struct _step_param
{
    int KernelStep;
    int OutputStep;
};
struct _step_param2
{
    int KernelStep_w;
    int KernelStep_h;
    int OutputStep_w;
    int OutputStep_h;
};
enum LayerType{
         Convolution
        ,Pooling
};
class BaseLayer
{
    struct InnerContext
    {
        int w_in;
        int h_in;
        int w_out;
        int h_out;
        int channels_in;
        int channels_out;
        void (*set_rand)(target_type& v);

        //Pooling:
        int pooling_w;
        int pooling_h;
        int pooling_type; //0 - max, 1 - mean

        //Convolution:
        int kw;
        int kh;
        int padding_w;
        int padding_h;
        int step_w;
        int step_h;
        int stride_w;
        int stride_h;

        target_type learn_rate;
        ACTIVATION activation_type;

        bool use_common_kernel;
        bool use_act_signal;
    };

public:
    BaseLayer(cl_ConvolutionContext c, OCLContext *ocl, int MultiObjects = 1);
    BaseLayer(cl_PoolingContext c, OCLContext *ocl, int MultiObjects = 1);

    /*
    void forward_propagation() override;
    void alloc_output() override;
    void build_up() override;
    void setRandInput() override;
    void setInput(const DMultiMatrix<target_type> &data) override;
    void setInput(const target_type *data, int size = 0) override;
    void loadHostBuffer(LayerHostBuffers buf) override;
    DMultiMatrix<target_type> getAllChannels(LayerHostBuffers t) override;
    DMatrix<target_type> getChannel(LayerHostBuffers t, int ch) override;
    void showAllChannels(LayerHostBuffers t, const char *message = nullptr) override;

    void setRandWeights();
private:
    PHOSTBUFFER _getHostBuffer(LayerHostBuffers t) override;
    PDEVICEBUFFER _getDeviceBuffer(LayerHostBuffers t) override;
    int _getHostChannelsNumber(LayerHostBuffers t) override;
    void allocHostBuffer(LayerHostBuffers t) override;
    */

public:
    void forward_propagation();
    void back_propagation();
    void alloc_output();
    void build_up();
    void setRandInput();
    void setRandWeights();
    void setNormalizedWeights();
    void setInput(const DMultiMatrix<target_type> &data);
    void setInput(const target_type *data, int size = 0);
    void setInput(const DMultiMatrix<target_type> &data, int multiIndex);
    void connectNext(BaseLayer &next);
    int getOutWidth() const;
    int getOutHeight() const;
    int getInputChannelsNumber() const;
    int getOutputChannelsNumber() const;

    int getInputChannelArea() const;
    int getOutputChannelArea() const;
    int getInputTotalArea() const;
    int getOutputTotalArea() const;

    bool isConvolutionLayer() const;
    bool isPoolingLayer() const;


    void loadHostBuffer(LayerBufferType buf);
    DMultiMatrix<target_type> getAllChannels(LayerBufferType t);
    DMatrix<target_type> getChannel(LayerBufferType t, int ch);
    void showAllChannels(LayerBufferType t, const char *message = nullptr);

    int getDeviceMem() const;
    const char *getDeviceMemShortSize();
    int getHostMem() const;
    const char *getHostMemShortSize();
    bool check(int log_level, BaseLayer *next = nullptr);

    InnerContext context;


    int InputChannelArea;
    int OutputChannelArea;
    int KernelArea;
    int kernels_n;

    int InputDataSize;
    int OutputDataSize;
    int KernelDataSize;

    //------------------------------------
    DEVICEBUFFER _input_signal;
    DEVICEBUFFER _input_raw_signal;
    DEVICEBUFFER _input_error;
    DEVICEBUFFER _output_deract;
    DEVICEBUFFER _output_maxpooling_trace;
    DEVICEBUFFER _back_prop_param_track;

    DEVICEBUFFER _kernel;
    DEVICEBUFFER _bias;
    DEVICEBUFFER _kernel_delta;
    DEVICEBUFFER _bias_delta;


    PDEVICEBUFFER _ptr_output_signal;
    PDEVICEBUFFER _ptr_output_raw_signal;
    PDEVICEBUFFER _ptr_output_error;
    //------------------------------------
    HOSTBUFFER _hostInput;
    HOSTBUFFER _hostRawInput;
    HOSTBUFFER _hostInputError;
    HOSTBUFFER _hostDeractOutput;
    HOSTBUFFER _hostKernels;
    HOSTBUFFER _hostOutput_ref;
    HOSTBUFFER _hostRawOutput_ref;
    HOSTBUFFER _hostOutputError_ref;
    HOSTBUFFER _hostKernelsDelta;
    HOSTBUFFER _hostBias;
    HOSTBUFFER _hostBiasDelta;
    //--------------------------------------

    typedef cl::Kernel KERNEL_T;
    KERNEL_T _cl_kernel_main;
    KERNEL_T _cl_kernel_bias_out;
    KERNEL_T _cl_kernel_activation1;
    KERNEL_T _cl_kernel_backp;
    KERNEL_T _cl_kernel_apply_derivative2error;
    KERNEL_T _cl_kernel_make_delta;
    KERNEL_T _cl_kernel_make_bias_delta;
    KERNEL_T _cl_kernel_flush_delta;
    KERNEL_T _cl_kernel_flush_bias_delta;

    _step_param2 _back_prop_step_params;
private:
    OCLContext *ocl;
    cl::CommandQueue MainQueue;
    int DeviceMem;
    char deviceMemShortSize[20];
    int HostMem;
    char hostMemShortSize[20];
    bool is_built;
    int objects;



    LayerType LT;
    friend class cl_ConvNet;
private:
    void allocDeviceBuffer(DEVICEBUFFER &buffer, int size, int CellSize = 0);
    void allocDeviceBuffer(PDEVICEBUFFER &buffer, int size, int CellSize = 0);
    void allocHostBuffer(HOSTBUFFER &buffer, int channels_n, int w, int h);
    void allocHostBuffer(PHOSTBUFFER &buffer, int channels_n, int w, int h);
    void allocHostBuffer(LayerBufferType t);


    void copyToHostBuffer(const DEVICEBUFFER &DevBuffer, target_type *HostBuffer, int size);
    void copyToHostBuffer(const DEVICEBUFFER &DevBuffer, PHOSTBUFFER HostBuffer);
    void copyToDeviceBuffer(DEVICEBUFFER &DevBuffer, const target_type *HostBuffer, int size, int offset = 0);
    void copyToDeviceBuffer(DEVICEBUFFER &DevBuffer, const PHOSTBUFFER HostBuffer);
    void copyToDeviceBuffer(DEVICEBUFFER &DevBuffer, const void *HostData, int DataLenght);




    PHOSTBUFFER _getHostBuffer(LayerBufferType t);
    PDEVICEBUFFER _getDeviceBuffer(LayerBufferType t);
    int _getHostChannelsNumber(LayerBufferType t) const;
    const char *_getBufferName(LayerBufferType t);
};


int __cl_init(OCLContext &ocl, const char *program_path, int log_level = 0);

int __cd(int v1, int v2);
int __get_steps(int step, int stride, int pos);
_back_conv_param __getBackConvParams(int kernel_size, int input_size, int step, int stride, int padding, int input_pos);
_step_param __getBackConvSteps(int step, int stride);

//--------------------------
class cl_ConvNet
{
public:

    DArray<BaseLayer*> way;
    NET_P fnet;
    OCLContext ocl;
    int objects;


    bool is_init;
    bool is_build;

public:
    cl_ConvNet();
    BaseLayer* firstLayer() const;
    BaseLayer* lastLayer() const;
    BaseLayer *getLayer(int i) const;
    void AddLayer(cl_ConvolutionContext c);
    void AddLayer(cl_PoolingContext c);
    void AddFnLayer(int size);
    void InitOpenCL(const char *program_path);
    void setInput(const DMultiMatrix<target_type> &in);
    void setInput(const DMultiMatrix<target_type> &in, int MultiIndex);
    void setRandInput();
    void allocOutput();
    void buildUp();
    int baseSize() const;
    int baseInputChannels(int layer) const;
    int baseOutputChannels(int layer) const;
    void setRandWeights();
    void setNormalizedWeights();
    target_type* getFnetOutput(int &size);
    target_type* getFnetRawOutput(int &size);
    void useMultiObjectsProcessing(int ParallelObjects);

    NET_P getFullNet();
    bool check(int log_level);


    void forwardPropagation();
    void backPropagation();
    void transportDataToFullNet();
    void transportDataFromFullNet();
private:

};


#endif // OCL_LAYER_H
