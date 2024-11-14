#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "rknn_api.h"
#include <cstring>

#define DIM1 32
#define DIM2 15000

int main(int argc, char *argv[])
{
    /*要求程序传入的第一个参数为RKNN模型，第二个参数为要推理的图片*/
    char *model_path = argv[1];

    // 初始化随机数生成器
    srand((unsigned int)time(NULL));

    // 分配内存并生成随机数据
    float* data = (float*)malloc(DIM1 * DIM2 * sizeof(float));
    if (data == NULL) {
        printf("Memory allocation failed\n");
        return -1;
    }
    for (int i = 0; i < DIM1 * DIM2; i++) {
        data[i] = (float)rand() / RAND_MAX;  // 生成0到1之间的随机浮点数
    }

    /*调用rknn_init接口将RKNN模型的运行环境和相关信息赋予到context变量中*/
    rknn_context context;
    rknn_init(&context, model_path, 0, 0, NULL);

    /*调用rknn_query接口查询tensor输入输出个数*/
    rknn_input_output_num io_num;
    rknn_query(context, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    printf("model input num:%d,output num:%d\n", io_num.n_input, io_num.n_output);

    /*调用rknn_inputs_set接口设置输入数据*/
    rknn_input input[io_num.n_input];
    memset(input, 0, sizeof(rknn_input));
    input[0].index = 0;
    input[0].buf = data;
    input[0].size = DIM1 * DIM2 * sizeof(uint8_t);
    input[0].pass_through = 0;
    input[0].type = RKNN_TENSOR_FLOAT32;
    input[0].fmt = RKNN_TENSOR_UNDEFINED;
    rknn_inputs_set(context, io_num.n_input, input);

    /*调用rknn_run接口进行模型推理*/
    rknn_run(context, NULL);
    /*调用rknn_outputs_get接口获取模型推理结果*/
    rknn_output output[io_num.n_output];
    memset(output, 0, sizeof(rknn_output));
    output[0].index = 0;
    output[0].is_prealloc = 0;
    output[0].want_float = 1; // 表示将输出数据转换为浮点类型
    rknn_outputs_get(context, io_num.n_output, output, NULL);

    // 打印输出数据
    printf("Output tensor size: %d\n", output[0].size);
    // for (int i = 0; i < 10; i++) {
    //     printf("%f ", output[0].buf);
    // }
    // printf("\n");
    
    /*调用rknn_outputs_release接口释放推理输出的相关资源*/
    rknn_outputs_release(context, io_num.n_output, output);

    /*调用rknn_destory接口销毁context变量*/
    rknn_destroy(context);

    return 0;
}
