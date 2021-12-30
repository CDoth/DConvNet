struct mpair{int w; int h;};

/// -----------------------------------

void kernel set_bias
(
        global float *OUTPUT
        ,global float *BIAS
        ,int Output__ChannelArea
        )
{
    const int global_id = get_global_id(0);
    const int channel_id = global_id / Output__ChannelArea;
    OUTPUT[global_id] = BIAS[channel_id];

//    printf(" !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! set_bias: gid: %d bias: %f\n", global_id, BIAS[channel_id]);
}
void kernel set_bias_multiobject
(
        global float *OUTPUT
        ,global float *BIAS
        ,const int Output__ChannelArea
        ,const int Output__ChannelsNumber
        )
{
    const int global_id = get_global_id(0);
    const int channel_id = global_id / Output__ChannelArea;
    OUTPUT[global_id] = BIAS[channel_id % Output__ChannelsNumber];
}
/// ----------------------------------- CONVOLUTION
void kernel convolution_1_int
(
        global int *INPUT, global int *KERNEL, global int *OUT,
        int kw, int kh, int OutArea,  int InputArea, int KernelArea,
        int InputSize, int InputW, int InputH, int OutW, int OutH
        ,int paddingW, int paddingH
        ,int step_w, int step_h
        ,int stride_w, int stride_h
        )
{
    int global_id = get_global_id(0);
    int out_index = global_id % OutArea;
    int out_channel_id = global_id / OutArea;
    int out_w_index = out_index / OutH;
    int out_h_index = out_index % OutH;

    const struct mpair vksize = {(kw-1)*stride_w + kw, (kh-1)*stride_h + kh};
    struct mpair mul;
    struct mpair pad = {step_w * out_w_index - paddingW, step_h * out_h_index - paddingH};
    struct mpair pos;
    struct mpair shift;
    struct mpair cross_area;
    struct mpair left_blocks;
    struct mpair right_blocks;
    struct mpair local_pos;
    const struct mpair lim = {InputW - vksize.w, InputH - vksize.h};

    pos.w = pad.w < 0 ? 0 : pad.w;
    pos.h = pad.h < 0 ? 0 : pad.h;
    shift.w = (pad.w - pos.w) * -1;
    shift.h = (pad.h - pos.h) * -1;
    cross_area.w = pad.w < 0 ? vksize.w + pad.w : vksize.w;
    cross_area.w = pos.w > lim.w ? vksize.w - (pos.w - lim.w) : cross_area.w;

    cross_area.h = pad.h < 0 ? vksize.h + pad.h : vksize.h;
    cross_area.h = pos.h > lim.h ? vksize.h - (pos.h - lim.h) : cross_area.h;

    mul.w = cross_area.w / (stride_w+1);
    left_blocks.w = shift.w % (stride_w+1);
    right_blocks.w = cross_area.w % (stride_w+1);
    if(right_blocks.w > 0 && left_blocks.w <= right_blocks.w) ++mul.w;

    mul.h = cross_area.h / (stride_h+1);
    left_blocks.h = shift.h % (stride_h+1);
    right_blocks.h = cross_area.h % (stride_h+1);
    if(right_blocks.h > 0 && left_blocks.h <= right_blocks.h) ++mul.h;

    shift.w /= (stride_w+1);
    if(left_blocks.w > 0) ++shift.w;
    shift.h /= (stride_h+1);
    if(left_blocks.h > 0) ++shift.h;


    const int _kernel_matrix_shift = out_channel_id * InputSize;
    for(int INPUT_I = 0; INPUT_I != InputSize; ++INPUT_I)
    {
        for(int i=0;i!=mul.w;++i)
        {
            local_pos.w = i ? local_pos.w + stride_w+1 : left_blocks.w + pos.w;
            for(int j=0;j!=mul.h;++j)
            {
                local_pos.h = j ? local_pos.h + stride_h+1 : left_blocks.h + pos.h;
                        OUT[global_id] +=

                        INPUT
                        [
                            (InputArea * INPUT_I ) +  ( (local_pos.w) * InputH + (local_pos.h) )
                        ]
                        *
                        KERNEL
                        [
                            (_kernel_matrix_shift + INPUT_I) * KernelArea + ((i+shift.w) * kh + (j+shift.h))
                        ]
                        ;
            }
        }
    }
}

void kernel convolution_1_float
(
        global float *INPUT, global float *KERNEL, global float *OUT,
        int kw, int kh,
        int OutArea, int InputArea, int KernelArea,
        int InputSize, int InputW, int InputH, int OutW, int OutH
        ,int paddingW, int paddingH
        ,int step_w, int step_h
        ,int stride_w, int stride_h
        )
{
    int global_id = get_global_id(0);
    int out_index = global_id % OutArea;
    int out_channel_id = global_id / OutArea;
    int out_w_index = out_index / OutH;
    int out_h_index = out_index % OutH;

    const struct mpair vksize = {(kw-1)*stride_w + kw, (kh-1)*stride_h + kh};
    struct mpair mul;
    struct mpair pad = {step_w * out_w_index - paddingW, step_h * out_h_index - paddingH};
    struct mpair pos;
    struct mpair shift;
    struct mpair cross_area;
    struct mpair left_blocks;
    struct mpair right_blocks;
    struct mpair local_pos;
    const struct mpair lim = {InputW - vksize.w, InputH - vksize.h};

    pos.w = pad.w < 0 ? 0 : pad.w;
    pos.h = pad.h < 0 ? 0 : pad.h;
    shift.w = (pad.w - pos.w) * -1;
    shift.h = (pad.h - pos.h) * -1;
    cross_area.w = pad.w < 0 ? vksize.w + pad.w : vksize.w;
    cross_area.w = pos.w > lim.w ? vksize.w - (pos.w - lim.w) : cross_area.w;

    cross_area.h = pad.h < 0 ? vksize.h + pad.h : vksize.h;
    cross_area.h = pos.h > lim.h ? vksize.h - (pos.h - lim.h) : cross_area.h;

    mul.w = cross_area.w / (stride_w+1);
    left_blocks.w = shift.w % (stride_w+1);
    right_blocks.w = cross_area.w % (stride_w+1);
    if(right_blocks.w > 0 && left_blocks.w <= right_blocks.w) ++mul.w;

    mul.h = cross_area.h / (stride_h+1);
    left_blocks.h = shift.h % (stride_h+1);
    right_blocks.h = cross_area.h % (stride_h+1);
    if(right_blocks.h > 0 && left_blocks.h <= right_blocks.h) ++mul.h;

    shift.w /= (stride_w+1);
    if(left_blocks.w > 0) ++shift.w;
    shift.h /= (stride_h+1);
    if(left_blocks.h > 0) ++shift.h;


    const int _kernel_matrix_shift = out_channel_id * InputSize;
    for(int INPUT_I = 0; INPUT_I != InputSize; ++INPUT_I)
    {
        for(int i=0;i!=mul.w;++i)
        {
            local_pos.w = i ? local_pos.w + stride_w+1 : left_blocks.w + pos.w;
            for(int j=0;j!=mul.h;++j)
            {
                local_pos.h = j ? local_pos.h + stride_h+1 : left_blocks.h + pos.h;
                        OUT[global_id] +=

                        INPUT
                        [
                            (INPUT_I * InputArea)  +  ( (local_pos.w) * InputH + (local_pos.h) )
                        ]
                        *
                        KERNEL
                        [
                            (_kernel_matrix_shift + INPUT_I) * KernelArea + ((i+shift.w) * kh + (j+shift.h))
                        ]
                        ;
            }
        }
    }
}
void kernel convolution_1_float_multiobject
(
        global float *INPUT, global float *KERNEL, global float *OUT,
        int kw, int kh,
        int ObjectOutDataSize,
        int OutArea, int InputArea, int KernelArea,
        int InputSize, int InputW, int InputH, int OutW, int OutH
        ,int paddingW, int paddingH
        ,int step_w, int step_h
        ,int stride_w, int stride_h
        )
{
    const int global_id = get_global_id(0);

    const int object_index = global_id / ObjectOutDataSize;
    const int local_out_index = global_id % ObjectOutDataSize;
    const int out_channel_id = local_out_index / OutArea;
    const int out_index = local_out_index % OutArea;

    const int out_w_index = out_index / OutH;
    const int out_h_index = out_index % OutH;

    const struct mpair vksize = {(kw-1)*stride_w + kw, (kh-1)*stride_h + kh};
    struct mpair mul;
    struct mpair pad = {step_w * out_w_index - paddingW, step_h * out_h_index - paddingH};
    struct mpair pos;
    struct mpair shift;
    struct mpair cross_area;
    struct mpair left_blocks;
    struct mpair right_blocks;
    struct mpair local_pos;
    const struct mpair lim = {InputW - vksize.w, InputH - vksize.h};

    pos.w = pad.w < 0 ? 0 : pad.w;
    pos.h = pad.h < 0 ? 0 : pad.h;
    shift.w = (pad.w - pos.w) * -1;
    shift.h = (pad.h - pos.h) * -1;
    cross_area.w = pad.w < 0 ? vksize.w + pad.w : vksize.w;
    cross_area.w = pos.w > lim.w ? vksize.w - (pos.w - lim.w) : cross_area.w;

    cross_area.h = pad.h < 0 ? vksize.h + pad.h : vksize.h;
    cross_area.h = pos.h > lim.h ? vksize.h - (pos.h - lim.h) : cross_area.h;

    mul.w = cross_area.w / (stride_w+1);
    left_blocks.w = shift.w % (stride_w+1);
    right_blocks.w = cross_area.w % (stride_w+1);
    if(right_blocks.w > 0 && left_blocks.w <= right_blocks.w) ++mul.w;

    mul.h = cross_area.h / (stride_h+1);
    left_blocks.h = shift.h % (stride_h+1);
    right_blocks.h = cross_area.h % (stride_h+1);
    if(right_blocks.h > 0 && left_blocks.h <= right_blocks.h) ++mul.h;

    shift.w /= (stride_w+1);
    if(left_blocks.w > 0) ++shift.w;
    shift.h /= (stride_h+1);
    if(left_blocks.h > 0) ++shift.h;


    const int _kernel_matrix_shift = out_channel_id * InputSize;
    const int _input_matrix_shift = object_index * InputSize;
    for(int INPUT_I = 0; INPUT_I != InputSize; ++INPUT_I)
    {
        for(int i=0;i!=mul.w;++i)
        {
            local_pos.w = i ? local_pos.w + stride_w+1 : left_blocks.w + pos.w;
            for(int j=0;j!=mul.h;++j)
            {
                local_pos.h = j ? local_pos.h + stride_h+1 : left_blocks.h + pos.h;
                        OUT[global_id] +=

                        INPUT
                        [
                            (_input_matrix_shift + INPUT_I) * InputArea  +  ( (local_pos.w) * InputH + (local_pos.h) )
                        ]
                        *
                        KERNEL
                        [
                            (_kernel_matrix_shift + INPUT_I) * KernelArea + ((i+shift.w) * kh + (j+shift.h))
                        ]
                        ;
            }
        }
    }
}
void kernel convolution_2_float
(
        global float *INPUT, global float *KERNEL, global float *OUT,
        int kw, int kh,
        int OutArea, int InputArea, int KernelArea,
        int InputSize,
        int InputW, int InputH,
        int OutW, int OutH
        ,int step_w, int step_h
        )
{
    int global_id = get_global_id(0);
    int out_index = global_id % OutArea;
    int out_channel_id = global_id / OutArea;
    int out_w_index = out_index / OutH;
    int out_h_index = out_index % OutH;

    int input_w_index = out_w_index * step_w;
    int input_h_index = out_h_index * step_h;



    const int _kernel_matrix_shift = out_channel_id * InputSize;

    OUT[global_id] = 0.0f;

    for(int INPUT_I = 0; INPUT_I != InputSize; ++INPUT_I)
    {
        for(int i=0;i!=kw;++i)
        {
            for(int j=0;j!=kh;++j)
            {
                        OUT[global_id] +=

                        INPUT
                        [
                            (InputArea * INPUT_I ) +  ( (input_w_index+i) * InputH + (input_h_index + j) )
                        ]
                        *
                        KERNEL
                        [
                            (_kernel_matrix_shift + INPUT_I) * KernelArea + (i * kh + j)
                        ]
                        ;
            }
        }
    }
}
/// ----------------------------------- BACK CONVOLUTION
struct BackConvParam
{
    int KernelStartPoint_w;
    int OutputStartPoint_w;
    int n_w;

    int KernelStartPoint_h;
    int OutputStartPoint_h;
    int n_h;
};
struct StepParam
{
    int KernelStep_w;
    int KernelStep_h;
    int OutputStep_w;
    int OutputStep_h;
};

void kernel back_convolution_1_int(
        global int *INPUT_ERROR, global int *KERNEL, global int *OUTPUT_ERROR
        ,int InputSize ,int OutputSize
        ,int InputArea, int KW, int KH, int OutH

        ,int InputW,int InputH
        ,struct StepParam steps
        ,global struct BackConvParam *param_track
        )
{
    int global_id = get_global_id(0);
    int InputChannelIndex = global_id / InputArea;
    int inner_index = global_id % InputArea;

    struct BackConvParam BCP = param_track[inner_index];
    int KernelIndex_w = 0;
    int KernelIndex_h = 0;
    int OutputIndex_w = 0;
    int OutputIndex_h = 0;


//    printf("test_back_prop_1: META: gid: %d ii: %d ih: %d "
//           "[W: p: %d Track: (OS: %d) (KS: %d) (N: %d) ::::: Calc: (OS: %d) (KS: %d) (N: %d)]  "
//           "[H: p: %d Track: (OS: %d) (KS: %d) (N: %d) ::::: Calc: (OS: %d) (KS: %d) (N: %d)]  "
//           "\n",
//           global_id, inner_index, InputH,
//           POS.w, BCP.OutputStartPoint_w, BCP.KernelStartPoint_w, BCP.n_w,    OUTPUT_START.w, KERNEL_START.w, N.w,
//           POS.h, BCP.OutputStartPoint_h, BCP.KernelStartPoint_h, BCP.n_h,    OUTPUT_START.h, KERNEL_START.h, N.h
//           );


    for(int OUT_I=0;OUT_I!=OutputSize;++OUT_I)
    {
        KernelIndex_w = BCP.KernelStartPoint_w;
        OutputIndex_w = BCP.OutputStartPoint_w;
        for(int i=0;i!=BCP.n_w;++i)
        {
            KernelIndex_h = BCP.KernelStartPoint_h;
            OutputIndex_h = BCP.OutputStartPoint_h;
            for(int j=0;j!=BCP.n_h;++j)
            {
                INPUT_ERROR[global_id] +=
                        KERNEL
                        [
                            (OUT_I * InputSize + InputChannelIndex)  +  (KernelIndex_w * KH + KernelIndex_h)
                        ]
                        *
                        OUTPUT_ERROR
                        [
                            OUT_I + (OutputIndex_w * OutH + OutputIndex_h)
                        ]
                        ;


//                printf("test_back_prop_1: global_id: %d KERNEL: %d OUTPUT_ERROR: %d  "
//                       "KernelIndex_h: %d OutputIndex_h: %d KernelIndex_w: %d OutputIndex_w: %d"
//                       "\n", global_id
//                       ,
//                       KERNEL
//                       [
//                       (OUT_I * InputSize + InputChannelIndex)  +  (KernelIndex_w * KH + KernelIndex_h)
//                       ]
//                        ,
//                        OUTPUT_ERROR
//                        [
//                        OUT_I + (OutputIndex_w * OutH + OutputIndex_h)
//                        ]
//                        ,KernelIndex_h, OutputIndex_h, KernelIndex_w, OutputIndex_w
//                        );

                KernelIndex_h -= steps.KernelStep_h;
                OutputIndex_h += steps.OutputStep_h;
            }
            KernelIndex_w -= steps.KernelStep_w;
            OutputIndex_w += steps.OutputStep_w;
        }
    }
}
void kernel back_convolution_1_float(

        global float *INPUT_ERROR, global float *KERNEL, global float *OUTPUT_ERROR

        ,int Input__ChannelArea
        ,int Input__ChannelsNumber

        ,int Output__Height
        ,int Output__ChannelArea
        ,int Output__ChannelsNumber

        ,int Kernel__Area
        ,int Kernel__Height

        ,struct StepParam steps
        ,global struct BackConvParam *param_track
        )
{
    int global_id = get_global_id(0);
    //----------------------------------------
    int in_channel_id = global_id / Input__ChannelArea;
    int in_index = global_id % Input__ChannelArea;
    //----------------------------------------

    struct BackConvParam BCP = param_track[in_index];
    int KernelIndex_w = 0;
    int KernelIndex_h = 0;
    int OutputIndex_w = 0;
    int OutputIndex_h = 0;

//    printf("back_convolution_1_float: gid: %d in_index: %d nw: %d nh: %d\n", global_id, in_index, BCP.n_w, BCP.n_h);


    int kernel_index = 0;
    int output_index = 0;

    for(int OUT_I=0;OUT_I!=Output__ChannelsNumber;++OUT_I)
    {
        KernelIndex_w = BCP.KernelStartPoint_w;
        OutputIndex_w = BCP.OutputStartPoint_w;
        for(int i=0;i!=BCP.n_w;++i)
        {
            KernelIndex_h = BCP.KernelStartPoint_h;
            OutputIndex_h = BCP.OutputStartPoint_h;
            for(int j=0;j!=BCP.n_h;++j)
            {
                kernel_index = (OUT_I * Input__ChannelsNumber + in_channel_id) * Kernel__Area + (KernelIndex_w * Kernel__Height + KernelIndex_h);
                output_index = (OUT_I * Output__ChannelArea) + (OutputIndex_w * Output__Height + OutputIndex_h);
                INPUT_ERROR[global_id] +=
                        OUTPUT_ERROR
                        [
                            output_index
                        ]
                        *
                        KERNEL
                        [
                            kernel_index
                        ]
                        ;

                KernelIndex_h -= steps.KernelStep_h;
                OutputIndex_h += steps.OutputStep_h;
            }
            KernelIndex_w -= steps.KernelStep_w;
            OutputIndex_w += steps.OutputStep_w;
        }
    }

}
void kernel back_convolution_1_float_multiobject(

        global float *INPUT_ERROR,  global float *KERNEL, global float *OUTPUT_ERROR

        ,int Input__TotalArea
        ,int Input__ChannelArea
        ,int Input__ChannelsNumber

        ,int Output__Height
        ,int Output__ChannelArea
        ,int Output__ChannelsNumber

        ,int Kernel__Area
        ,int Kernel__Height

        ,struct StepParam steps
        ,global struct BackConvParam *param_track
        )
{
    int global_id = get_global_id(0);

    //----------------------------------------
    const int object_index = global_id / Input__TotalArea;
    const int local_in_index = global_id % Input__TotalArea;
    const int in_channel_id = local_in_index / Input__ChannelArea;
    const int in_index = local_in_index % Input__ChannelArea;
    //----------------------------------------

    struct BackConvParam BCP = param_track[in_index];
    int KernelIndex_w = 0;
    int KernelIndex_h = 0;
    int OutputIndex_w = 0;
    int OutputIndex_h = 0;

    const int _output_object_shift = object_index * Output__ChannelsNumber;
    int kernel_index = 0;
    int output_index = 0;
    for(int OUT_I=0;OUT_I!=Output__ChannelsNumber;++OUT_I)
    {
        KernelIndex_w = BCP.KernelStartPoint_w;
        OutputIndex_w = BCP.OutputStartPoint_w;
        for(int i=0;i!=BCP.n_w;++i)
        {
            KernelIndex_h = BCP.KernelStartPoint_h;
            OutputIndex_h = BCP.OutputStartPoint_h;
            for(int j=0;j!=BCP.n_h;++j)
            {
                kernel_index = (OUT_I * Input__ChannelsNumber + in_channel_id) * Kernel__Area + (KernelIndex_w * Kernel__Height + KernelIndex_h);
                output_index = (OUT_I + _output_object_shift) * Output__ChannelArea + (OutputIndex_w * Output__Height + OutputIndex_h);

                INPUT_ERROR[global_id] +=
                        OUTPUT_ERROR
                        [
                            output_index
                        ]
                        *
                        KERNEL
                        [
                            kernel_index
                        ]
                        ;

                KernelIndex_h -= steps.KernelStep_h;
                OutputIndex_h += steps.OutputStep_h;
            }
            KernelIndex_w -= steps.KernelStep_w;
            OutputIndex_w += steps.OutputStep_w;
        }
    }
}
void kernel back_convolution_2_int(
        global int *INPUT_ERROR, global int *KERNEL, global int *OUTPUT_ERROR, global int *KERNEL_DELTA, global int *INPUT_SIGNAL, int learn_rate
        ,int InputSize ,int OutputSize
        ,int InputArea, int KW, int KH, int OutH

        ,int InputW,int InputH
        ,struct StepParam steps
        ,global struct BackConvParam *param_track
        )
{
    int global_id = get_global_id(0);
    int InputChannelIndex = global_id / InputArea;
    int inner_index = global_id % InputArea;

    struct BackConvParam BCP = param_track[inner_index];
    int KernelIndex_w = 0;
    int KernelIndex_h = 0;
    int OutputIndex_w = 0;
    int OutputIndex_h = 0;




    int _kernel_index = 0;
    int _output_index = 0;
    for(int OUT_I=0;OUT_I!=OutputSize;++OUT_I)
    {
        KernelIndex_w = BCP.KernelStartPoint_w;
        OutputIndex_w = BCP.OutputStartPoint_w;
        for(int i=0;i!=BCP.n_w;++i)
        {
            KernelIndex_h = BCP.KernelStartPoint_h;
            OutputIndex_h = BCP.OutputStartPoint_h;
            for(int j=0;j!=BCP.n_h;++j)
            {
                _kernel_index = (OUT_I * InputSize + InputChannelIndex)  +  (KernelIndex_w * KH + KernelIndex_h);
                _output_index = OUT_I + (OutputIndex_w * OutH + OutputIndex_h);

                INPUT_ERROR
                [
                    global_id
                ]
                        +=
                        KERNEL
                        [
                            _kernel_index
                        ]
                        *
                        OUTPUT_ERROR
                        [
                            _output_index
                        ]
                        ;

                KERNEL_DELTA
                [
                     _kernel_index
                ]
                        +=
                        OUTPUT_ERROR
                        [
                            _output_index
                        ]
                        *
                        INPUT_SIGNAL
                        [
                            global_id
                        ]
                        *
                        learn_rate
                        ;


                KernelIndex_h -= steps.KernelStep_h;
                OutputIndex_h += steps.OutputStep_h;
            }
            KernelIndex_w -= steps.KernelStep_w;
            OutputIndex_w += steps.OutputStep_w;
        }
    }
}

/// ----------------------------------- DELTAS
void kernel make_delta_1 //zero-padding version
(
         global const float *INPUT_SIGNAL
        ,global const float *OUTPUT_ERROR
        ,global float *DELTA
        ,const float learn_rate

        ,const int step_w
        ,const int step_h
        ,const int stride_w
        ,const int stride_h

        ,const int Output__ChannelArea
        ,const int Output__Width
        ,const int Output__Height

        ,const int Input__ChannelArea
        ,const int Input__ChannelsNumber
        ,const int Input__Height

        ,const int Kernel__Area
        ,const int Kernel__Height

        )
{
    const int global_id = get_global_id(0);

    const int kernel_id = global_id / Kernel__Area;
    const int weight_id = global_id % Kernel__Area;
    const int output_channel_id = kernel_id / Input__ChannelsNumber;
    const int input_channel_id = kernel_id % Input__ChannelsNumber;

    const int weight_w_pos = weight_id / Kernel__Height;
    const int weight_h_pos = weight_id % Kernel__Height;
    const int weight_w_virtual_pos = (weight_w_pos)*stride_w + weight_w_pos;
    const int weight_h_virtual_pos = (weight_h_pos)*stride_h + weight_h_pos;

    const int output_channel_shift = output_channel_id * Output__ChannelArea;
    const int input_channel_shift = input_channel_id * Input__ChannelArea;

    int input_w_index = 0;
    int input_h_index = 0;

    int input_index = 0;
    int output_index = 0;


    input_w_index = weight_w_virtual_pos;
    for(int output_w_index=0;output_w_index!=Output__Width;++output_w_index)
    {
        input_h_index = weight_h_virtual_pos;
        for(int output_h_index=0;output_h_index!=Output__Height;++output_h_index)
        {
            input_index = input_channel_shift + (input_w_index * Input__Height + input_h_index);
            output_index = output_channel_shift + (output_w_index * Output__Height + output_h_index);

            DELTA[global_id] +=

            INPUT_SIGNAL
                    [
                       input_index
                    ]
            *
            OUTPUT_ERROR
                    [
                       output_index
                    ]
            *
            learn_rate;


            input_h_index += step_h;
        }
        input_w_index += step_w;
    }


}
void kernel make_delta_1_multiobject //zero-padding version
(
         global const float *INPUT_SIGNAL
        ,global const float *OUTPUT_ERROR
        ,global float *DELTA
        ,const float learn_rate

        ,const int step_w
        ,const int step_h
        ,const int stride_w
        ,const int stride_h

        ,const int Output__ChannelArea
        ,const int Output__Width
        ,const int Output__Height
        ,const int Output__ChannelsNumber

        ,const int Input__ChannelArea
        ,const int Input__ChannelsNumber
        ,const int Input__Height

        ,const int Kernel__Area
        ,const int Kernel__Height

        ,const int Kernel__TotalArea
        ,const int Kernel__ChannelsNumber

        )
{
    const int global_id = get_global_id(0);

    const int object_index = global_id / Kernel__TotalArea;

    const int kernel_id = (global_id / Kernel__Area) % Kernel__ChannelsNumber;
    const int weight_id = global_id % Kernel__Area;

    const int output_channel_id = kernel_id / Input__ChannelsNumber;
    const int input_channel_id = kernel_id % Input__ChannelsNumber;

    const int weight_w_pos = weight_id / Kernel__Height;
    const int weight_h_pos = weight_id % Kernel__Height;
    const int weight_w_virtual_pos = (weight_w_pos)*stride_w + weight_w_pos;
    const int weight_h_virtual_pos = (weight_h_pos)*stride_h + weight_h_pos;

    const int output_object_shift = object_index * Output__ChannelsNumber * Output__ChannelArea;
    const int input_object_shift = object_index * Input__ChannelsNumber * Input__ChannelArea;

    const int output_channel_shift = output_channel_id * Output__ChannelArea;
    const int input_channel_shift = input_channel_id * Input__ChannelArea;

    int input_w_index = 0;
    int input_h_index = 0;

    int input_index = 0;
    int output_index = 0;
    input_w_index = weight_w_virtual_pos;
    for(int output_w_index=0;output_w_index!=Output__Width;++output_w_index)
    {
        input_h_index = weight_h_virtual_pos;
        for(int output_h_index=0;output_h_index!=Output__Height;++output_h_index)
        {
            input_index = input_object_shift + input_channel_shift + (input_w_index * Input__Height + input_h_index);
            output_index = output_object_shift + output_channel_shift + (output_w_index * Output__Height + output_h_index);

            DELTA[global_id] +=

            INPUT_SIGNAL
                    [
                       input_index
                    ]
            *
            OUTPUT_ERROR
                    [
                       output_index
                    ]
            *
            learn_rate;


            input_h_index += step_h;
        }
        input_w_index += step_w;
    }

}

void kernel make_bias_delta
(
        global const float *OUTPUT_ERROR
        ,global float *BIAS_DELTA
        ,const float learn_rate

        ,const int Output__ChannelArea
        )
{
    const int global_id = get_global_id(0);

    for(int i=0;i!=Output__ChannelArea; ++i)
    {
        BIAS_DELTA[global_id] +=
                OUTPUT_ERROR
                [
                    global_id * Output__ChannelArea + i
                ]
                *
                learn_rate;
    }
}
void kernel make_bias_delta_multiobject
(
        global const float *OUTPUT_ERROR
        ,global float *BIAS_DELTA
        ,const float learn_rate

        ,const int Output__ChannelArea
        ,const int Output__ChannelsNumber
        )
{
    const int global_id = get_global_id(0);

    for(int i=0;i!=Output__ChannelArea; ++i)
    {
        BIAS_DELTA[global_id] +=
                OUTPUT_ERROR
                [
                    (global_id % Output__ChannelsNumber) * Output__ChannelArea + i
                ]
                *
                learn_rate;
    }
}
/// ----------------------------------- MAX POOLING
void kernel max_pooling_1_int
(
        global int *input,
        global int *output
                          ,int pooling_w, int pooling_h
                          ,int OutArea
                          ,int OutH
                          ,int InputH
                          ,int InputArea
                          ,global int *trace
                         )
{
    int global_id = get_global_id(0);

    int channel_index = global_id / OutArea;
    int out_index = global_id % OutArea;
    int out_w_index = out_index / OutH;
    int out_h_index = out_index % OutH;

    int input_w_index = out_w_index * pooling_w;
    int input_h_index = out_h_index * pooling_h;

    global int *_input = input + channel_index * InputArea;
    int curr_index = input_w_index * InputH + input_h_index;
    int max = _input[curr_index];
    trace[global_id] = curr_index;


    for(int i=0;i!=pooling_w;++i)
    {
        for(int j=0;j!=pooling_h;++j)
        {
            curr_index = (input_w_index+i)*InputH + (input_h_index+j);
            if(_input[curr_index] > max)
            {
                max = _input[curr_index];
                trace[global_id] = curr_index;
            }
        }
    }
    output[global_id] = max;
}
/// ----------------------------------- BACK MAX POOLING
void kernel back_max_pooling_int
(
         global int *INPUT_ERROR
        ,global int *OUTPUT_ERROR
        ,global int *TRACE
        ,int OutArea
        ,int InputArea
        )
{
    int global_id = get_global_id(0);
    int channel_index = global_id / OutArea;
    INPUT_ERROR[channel_index * InputArea + TRACE[global_id]] = OUTPUT_ERROR[global_id];
}
/// ----------------------------------- MEAN POOLING
void kernel mean_pooling_int
(
         global int *INPUT
        ,global int *OUTPUT
        ,int pooling_w, int pooling_h
        ,int OutArea
        ,int OutH
        ,int InputH
        ,int InputArea
        )
{
    int global_id = get_global_id(0);
    int channel_index = global_id / OutArea;
    global int *_INPUT = INPUT + channel_index * InputArea;

    int input_w_index = ((global_id % OutArea) / OutH) * pooling_w;
    int input_h_index = ((global_id % OutArea) % OutH) * pooling_h;

    int mean = 0;
    int rate = pooling_w * pooling_h;
    for(int i=0;i!=pooling_w;++i)
    {
        for(int j=0;j!=pooling_h;++j)
        {
            mean += _INPUT[(input_w_index+i) * InputH + (input_h_index+j)];
        }
    }
    OUTPUT[global_id] = mean / rate;
}
/// ----------------------------------- BACK MEAN POOLING

void kernel back_mean_pooling_int
(
         global int *INPUT_ERROR
        ,global int *OUTPUT_ERROR
        ,int pooling_w, int pooling_h
        ,int OutH
        ,int InputH
        ,int OutArea
        ,int InputArea
        )
{
    int global_id = get_global_id(0);
    int channel_index = global_id / OutArea;
    global int *_INPUT = INPUT_ERROR + channel_index * InputArea;
    int input_w_index = ((global_id % OutArea) / OutH) * pooling_w;
    int input_h_index = ((global_id % OutArea) % OutH) * pooling_h;
    int rate = pooling_w * pooling_h;

    for(int i=0;i!=pooling_w;++i)
    {
        for(int j=0;j!=pooling_h;++j)
        {
            _INPUT[(input_w_index+i) * InputH + (input_h_index+j)] = OUTPUT_ERROR[global_id] / rate;
        }
    }
}

/// ----------------------------------- ACTIVATIONS

void kernel act2_sigmoid(
        global float *raw_out,
        global float *act_out,
        global float *deract_out
        )
{
    int global_id = get_global_id(0);
    act_out[global_id] = 1.0/(1.0+exp(-raw_out[global_id]));
    float e = exp(-raw_out[global_id]);
    deract_out[global_id] = e/((e+1.0)*(e+1.0));
}
void kernel act2_gtan(
        global float *raw_out,
        global float *act_out,
        global float *deract_out
        )
{
    int global_id = get_global_id(0);
    act_out[global_id] = (2.0/( 1.0 + exp(-raw_out[global_id]) )) - 1;
    float e = exp(-raw_out[global_id]);
    deract_out[global_id] = (2.0 * e)/((e+1.0)*(e+1.0));
}
void kernel act2_ReLU(
        global float *raw_out,
        global float *act_out,
        global float *deract_out
        )
{
    int global_id = get_global_id(0);
    act_out[global_id] = raw_out[global_id] > 0 ? raw_out[global_id] : 0;
    deract_out[global_id] = raw_out[global_id] > 0 ? 1 : 0;
}
void kernel act2_LINE(
        global float *raw_out,
        global float *act_out,
        global float *deract_out
        )
{
    int global_id = get_global_id(0);
    act_out[global_id] = raw_out[global_id];
    deract_out[global_id] = 1;
}
/// ----------------------------------- OTHER
void kernel multiply
(
         global float *src_buffer
        ,global float *dst_buffer
        )
{
    int global_id = get_global_id(0);
    dst_buffer[global_id] *= src_buffer[global_id];
}

void kernel zero
(
        global float *src_buffer
        )
{
    src_buffer[get_global_id(0)] = 0;
}
void kernel add
(
        global float *src_buffer
       ,global float *dst_buffer
        )
{
    int global_id = get_global_id(0);
    dst_buffer[global_id] += src_buffer[global_id];
}
void kernel flush
(
         global float *src_buffer
        ,global float *dst_buffer
        )
{
    int global_id = get_global_id(0);
    dst_buffer[global_id] += src_buffer[global_id];
    src_buffer[global_id] = 0;
}
void kernel fill
(
        global float *dst_buffer
        ,float value
        )
{
    dst_buffer[get_global_id(0)] = value;
}
