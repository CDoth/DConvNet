#ifndef LAYER_H
#define LAYER_H


typedef float value_type;
typedef value_type cnet_type;
class base_layer
{
protected:
    int w_in;
    int h_in;
    int w_out;
    int h_out;

    base_layer *next;
public:
    int out_w() const {return w_out;}
    int out_h() const {return h_out;}
    int in_w() const {return w_in;}
    int in_h() const {return h_in;}

    virtual int channels_in() const = 0;
    virtual int channels_out() const = 0;
    virtual int output_area() const = 0;

    base_layer() {}
    virtual void build_up() = 0;
    virtual void rand_input() = 0;
    virtual void rand_output() = 0;
    virtual void forward_propagation_mt() = 0;
    virtual void back_propagation_mt() = 0;
    virtual void forward_propagation_debug() = 0;
    virtual void back_propagation_debug() = 0;
    virtual void set_input(DMultiMatrix<value_type>& _input) = 0;
    virtual void alloc_output() = 0;


    virtual void show_in(const char *mess = nullptr) const = 0;
    virtual void show_out(const char *mess = nullptr) const = 0;
    virtual void show_in_error(const char *mess = nullptr) const = 0;
    virtual void show_out_error(const char *mess = nullptr) const = 0;
    virtual void show_raw_in(const char *mess = nullptr) const = 0;
    virtual void show_raw_out(const char *mess = nullptr) const = 0;

    virtual void connect_next(DMultiMatrix<value_type> &next_input, DMultiMatrix<value_type> &next_raw, DMultiMatrix<value_type> &next_error) = 0;
    virtual void connect_next(value_type *next_input_place, value_type *next_raw_place, value_type *next_error_place) = 0;

    virtual DMultiMatrix<value_type>& output() = 0;

};
struct ConvolutionContext
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

    bool common_kernel;
    bool bp_act_signal;
    int threads;

    value_type learn_rate;


    value_type (*act)(value_type x);
    value_type (*der_act)(value_type x);
    void (*set_rand)(value_type& v);
};
struct conv_task
{
    DMatrix<value_type> *signal_input;
    DMatrix<value_type> *raw_input;
    DMatrix<value_type> *signal_output_ref;
    DMatrix<value_type> *raw_output_ref;


    DMatrix<value_type> *kernel;
    DMatrix<value_type> *special_kernel;
    value_type *bias;

    DMatrix<value_type> *output_deract;
    DMatrix<value_type> *error_in;
    DMatrix<value_type> *error_out;

    int apply_delta;
};
struct ThreadContent
{
    DArray<conv_task> tasks;
    DDuplexMatrixTrack<value_type> track;
    DMatrixTrack<value_type> kernel_delta_track;
    DThreadHandler th;
    DMatrix<value_type> kernel_delta;
};


void start_forward_thread(ThreadContent *tc, ConvolutionContext *context);
void start_back_thread(ThreadContent *thc, ConvolutionContext *context);

void _change_bias(DArray<value_type> &bias, value_type learn_rate, const DMultiMatrix<value_type> &out_error);
class convolution_layer : public base_layer
{
public:
    DMultiMatrix<value_type> signal_input;
    DMultiMatrix<value_type> raw_input;

    DMultiMatrix<value_type> signal_output_ref;
    DMultiMatrix<value_type> raw_output_ref;



    DMultiMatrix<value_type> kernel;
    DMultiMatrix<value_type> special_kernel;
    DArray<value_type> bias;

    DMultiMatrix<value_type> output_deract; //real


    DMultiMatrix<cnet_type> error_in; //real
    DMultiMatrix<cnet_type> error_out_ref; //next error

    DArray<ThreadContent> plan;
    ConvolutionContext context;




public:

    int channels_in() const override {return context.channels_in;}
    int channels_out() const override {return context.channels_out;}
    int output_area() const override {return context.channels_out * w_out * h_out;}



    convolution_layer(ConvolutionContext _context, int alloc_input = 1);
    void build_up() override;
    void rand_input() override;
    void rand_output() override;
    void forward_propagation_mt() override;
    void back_propagation_mt() override;
    void forward_propagation_debug() override;
    void back_propagation_debug() override;
    void set_input(DMultiMatrix<value_type> &_input) override;
    void alloc_output() override;

    void show_in(const char *mess = nullptr) const override;
    void show_out(const char *mess = nullptr) const override;
    void show_in_error(const char *mess = nullptr) const override;
    void show_out_error(const char *mess = nullptr) const override;
    void show_raw_in(const char *mess = nullptr) const override;
    void show_raw_out(const char *mess = nullptr) const override;

    void connect_next(DMultiMatrix<value_type> &next_input, DMultiMatrix<value_type> &next_raw, DMultiMatrix<value_type> &next_error) override;
    void connect_next(value_type *next_input_place, value_type *next_raw_place, value_type *next_error_place) override;

    DMultiMatrix<value_type> & output() override;
};
//----------------------------------------------------------------------------------------------
struct PoolingContext
{
    int w;
    int h;
    int pw;
    int ph;
    int step_w;
    int step_h;
    int stride_w;
    int stride_h;
    int pooling_type; //0 - max, 1 - mean

    int channels;
    int threads;

    void (*set_rand)(value_type& v);
};
struct pooling_task
{
    DMatrix<complex_value_type<value_type>> *input;
    DMatrix<value_type> *error_input;
    DMatrix<value_type> *signal_output_ref;
    DMatrix<value_type> *raw_output_ref;
    DMatrix<value_type> *error_output_ref;

    DMatrix<value_type*> *error_trace;
};
struct PoolingThreadContent
{
    DArray<pooling_task> tasks;
    DMatrixTrack<complex_value_type<value_type>> track;
    DThreadHandler th;
};
void start_forward_pooling_thread(PoolingThreadContent *thc, PoolingContext *context);
void start_back_pooling_thread(PoolingThreadContent *thc, PoolingContext *context);

class pooling_layer : public base_layer
{
public:
    //------------------------------------------------------------------
    DMultiMatrix<value_type> signal_input;
    DMultiMatrix<value_type> raw_input;
    DMultiMatrix<value_type> error_input;
    //Use instead:
    DMultiMatrix<complex_value_type<value_type>> complex_input;
    //------------------------------------------------------------------

    //reference to next input:
    DMultiMatrix<value_type> signal_output_ref;
    DMultiMatrix<value_type> raw_output_ref;
    DMultiMatrix<value_type> error_output_ref;

    DMultiMatrix<value_type*> error_trace; //for max pooling



    pooling_layer(PoolingContext _context, int alloc_input = 1);
    DArray<PoolingThreadContent> plan;
    PoolingContext context;


    int channels_in() const override {return context.channels;}
    int channels_out() const override {return context.channels;}
    int output_area() const override {return context.channels * w_out * h_out;}

    void rand_input() override;
    void rand_output() override;
    void build_up() override;
    void forward_propagation_mt() override;
    void back_propagation_mt() override;
    void forward_propagation_debug() override;
    void back_propagation_debug() override;
    void set_input(DMultiMatrix<value_type> &_input) override;
    void alloc_output() override;


    void show_in(const char *mess = nullptr) const override;
    void show_out(const char *mess = nullptr) const override;
    void show_in_error(const char *mess = nullptr) const override;
    void show_out_error(const char *mess = nullptr) const override;
    void show_raw_in(const char *mess = nullptr) const override;
    void show_raw_out(const char *mess = nullptr) const override;

    void connect_next(DMultiMatrix<value_type> &next_input, DMultiMatrix<value_type> &next_raw, DMultiMatrix<value_type> &next_error) override;
    void connect_next(value_type *next_input_place, value_type *next_raw_place, value_type *next_error_place) override;

    DMultiMatrix<value_type> & output() override;
};
//----------------------------------
class ConvNet
{
public:
    ConvNet();
    DArray<base_layer*> way;
    net *n;
    void addLayer(ConvolutionContext);
    void addLayer(PoolingContext);
    void addFnLayer(int size);

    void setInput(DMultiMatrix<value_type> &input);
    void allocOutput();
    void build_up();
    void forward_propagation();
    void back_propagation();

};


#endif // LAYER_H
