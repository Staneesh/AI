#include <cstdio>
#include <stdint.h>
#include <math.h> // TODO(Blk): implement exp(x), rand()
#include <limits.h> // NOTE(Blk): UINT_MAX
#include <ctime> // NOTE(Blk): time(0) in srand();

typedef uint64_t u64;
typedef uint32_t u32;
typedef uint16_t u16; 
typedef uint8_t  u8;  
typedef int64_t  i64;  
typedef int32_t  i32;  
typedef int16_t  i16;  
typedef int8_t   i8;   
typedef float    r32;
typedef double   r64;

#define internal static
#define local_persist static
#define global_variable static

const u32 MAX_LAYER_NODE_COUNT = 100;
const u32 MAX_LAYERS_COUNT = 10;

#define DEBUG 1

struct Node
{
    r32 activation;
    r32 edges[MAX_LAYER_NODE_COUNT];
};

struct Layer
{
    Node nodes[MAX_LAYER_NODE_COUNT];
};

struct NeuralNetwork
{
    Layer layers[MAX_LAYERS_COUNT];
    r32 output[MAX_LAYER_NODE_COUNT];
    r32 errors[MAX_LAYER_NODE_COUNT];
    r32 averagedDerivatives[MAX_LAYERS_COUNT][MAX_LAYER_NODE_COUNT][MAX_LAYER_NODE_COUNT];
    
    i32 hiddenLayersCount;
    i32 nodesInEachLayer;
    r32 bias;
    i32 trainingIterations; 
    r32 learningRate;
};

r32 sigmoid(r32 x)
{
    return 1.0f / (1.0f + exp(-x));
}

internal void
initNeuralNetwork(NeuralNetwork *nn)
{
    for (int layerIndex = 0;
         layerIndex < nn->hiddenLayersCount + 1;
         ++layerIndex)
    {
        for (int nodeIndex = 0;
             nodeIndex < nn->nodesInEachLayer;
             ++nodeIndex)
        {
            Node& curNode = nn->layers[layerIndex].nodes[nodeIndex];
            
            curNode.activation = 0.0f;
            
            for (i32 edgeIndex = 0;
                 edgeIndex < nn->nodesInEachLayer;
                 ++edgeIndex)
            {
                curNode.edges[edgeIndex] = (r32)rand() / UINT_MAX;//0.5f;
            }
        }
    }
}

internal void
propagateForward(NeuralNetwork *nn)
{
    for (i32 layerIndex = 0;
         layerIndex < nn->hiddenLayersCount + 1;
         ++layerIndex)
    {
        for (int nodeIndex = 0;
             nodeIndex < nn->nodesInEachLayer;
             ++nodeIndex)
        {
            Node& curNode = nn->layers[layerIndex].nodes[nodeIndex];
            
            // NOTE(Blk): clear neighbors
            for (i32 edgeIndex = 0;
                 edgeIndex < nn->nodesInEachLayer;
                 ++edgeIndex)
            {
                r32 *neighbor;
                if (layerIndex == nn->hiddenLayersCount)
                {
                    neighbor = &nn->output[edgeIndex];
                    
                }
                else
                {
                    neighbor = &nn->layers[layerIndex + 1].nodes[edgeIndex].activation;
                }
                
                *neighbor = 0;
            }
            
            // NOTE(Blk): compute weighted sum
            for (i32 edgeIndex = 0;
                 edgeIndex < nn->nodesInEachLayer;
                 ++edgeIndex)
            {
                r32 *neighbor;
                if (layerIndex == nn->hiddenLayersCount)
                {
                    neighbor = &nn->output[edgeIndex];
                    
                }
                else
                {
                    neighbor = &nn->layers[layerIndex + 1].nodes[edgeIndex].activation;
                }
                
                *neighbor += curNode.activation * curNode.edges[edgeIndex];
            }
            
            // NOTE(Blk): activation and bias
            for (i32 edgeIndex = 0;
                 edgeIndex < nn->nodesInEachLayer;
                 ++edgeIndex)
            {
                r32 *neighbor;
                if (layerIndex == nn->hiddenLayersCount)
                {
                    neighbor = &nn->output[edgeIndex];
                    
                }
                else
                {
                    neighbor = &nn->layers[layerIndex + 1].nodes[edgeIndex].activation;
                }
                
                *neighbor = *neighbor + nn->bias;
            }
            
        }
    }
}

internal void
feedInput(NeuralNetwork *nn, i32 input)
{
    for (i32 i = 0; i < nn->nodesInEachLayer; ++i)
    {
        nn->layers[0].nodes[i].activation = input;
    }
}

internal void
calculateErrors(NeuralNetwork *nn, i32 trueOutput)
{
    for (i32 i = 0; i < nn->nodesInEachLayer; ++i)
    {
        nn->errors[i] = 2.0f * (trueOutput - nn->output[i]);
        
    }
}

internal void
updateErrors(NeuralNetwork *nn, i32 toWhichLayer)
{
    Layer *curLayer = &nn->layers[toWhichLayer];
    
    r32 nextErrors[MAX_LAYER_NODE_COUNT] = {};
    
    for (i32 nodeIndex = 0;
         nodeIndex < nn->nodesInEachLayer;
         ++nodeIndex)
    {
        Node *curNode = &curLayer->nodes[nodeIndex];
        
        for (i32 edgeIndex = 0;
             edgeIndex < nn->nodesInEachLayer;
             ++edgeIndex)
        {
            r32 *curEdge = &curNode->edges[edgeIndex];
            
            nextErrors[nodeIndex] += (r32)*curEdge * nn->errors[edgeIndex];
        }
    }
    
    for (i32 i = 0; i < nn->nodesInEachLayer; ++i)
    {
        nn->errors[i] = nextErrors[i];
    }
}

internal void
backpropagate(NeuralNetwork *nn)
{
    for (i32 layerIndex = nn->hiddenLayersCount;
         layerIndex >= 0;
         --layerIndex)
    {
        Layer *curLayer = &nn->layers[layerIndex];
        
        for (i32 nodeIndex = 0;
             nodeIndex < nn->nodesInEachLayer;
             ++nodeIndex)
        {
            Node *curNode = &curLayer->nodes[nodeIndex];
            
            for (i32 edgeIndex = 0;
                 edgeIndex < nn->nodesInEachLayer;
                 ++edgeIndex)
            {
                r32 *curEdge = &curNode->edges[edgeIndex];
                
                r32 der = curNode->activation * nn->errors[edgeIndex];
                
                *curEdge = *curEdge + nn->learningRate * der;
                
                nn->averagedDerivatives[layerIndex][nodeIndex][edgeIndex] -= der / nn->trainingIterations;
                
            }
        }
        
        updateErrors(nn, layerIndex);
    }
}

internal r32
getTotalError(NeuralNetwork *nn, i32 trueOutput)
{
    calculateErrors(nn, trueOutput);
    
    r32 res = 0.0f;
    
    for (i32 i = 0; i < nn->nodesInEachLayer; ++i)
    {
        res += nn->errors[i] * nn->errors[i];
    }
    
    res /= nn->nodesInEachLayer;
    
    return res * 100;
}

#if DEBUG
internal void
printNet(NeuralNetwork *nn)
{
    printf("Net:\n");
    
    for (i32 layer = 0; layer < nn->hiddenLayersCount + 1; ++layer)
    {
        printf("Layer no:%d\n", layer);
        for (i32 node = 0; node < nn->nodesInEachLayer; ++node)
        {
            printf("\tNode no:%d\n", node);
            for (i32 edge = 0; edge < nn->nodesInEachLayer; ++edge)
            {
                printf("\t\tEdge no:%d == %f\n", edge, nn->layers[layer].nodes[node].edges[edge]);
            }
        }
    }
}
#endif

internal void
train(NeuralNetwork *nn)
{
    i32 iterationIndex = 0;
    
    while(iterationIndex < nn->trainingIterations)
    {
        
        i32 input = rand() % 10;
        while(input == 3)
        {
            input = rand() % 10;
        }
        
        i32 output = 2*input;
        
        feedInput(nn, input);
        
        propagateForward(nn);
        calculateErrors(nn, output);
        backpropagate(nn);
        
        
        
        
        //printf("Test number:%d  in:%d  out:%d  totalError:%f%%\n", iterationIndex, input, output, getTotalError(nn, output));
        //printNet(nn);
        
        
        ++iterationIndex;
    }
}

internal void
setWeightsToAveraged(NeuralNetwork *nn)
{
    for (i32 layer = 0; layer < nn->hiddenLayersCount + 1; ++layer)
    {
        for (i32 node = 0; node < nn->nodesInEachLayer; ++node)
        {
            for (i32 edge = 0; edge < nn->nodesInEachLayer; ++edge)
            {
                nn->layers[layer].nodes[node].edges[edge] -= 
                    nn->averagedDerivatives[layer][node][edge] * nn->learningRate;
                nn->averagedDerivatives[layer][node][edge] = 0;
            }
        }
    }
    
}

int main()
{
    srand(time(0));
    
    NeuralNetwork net = {};
    net.hiddenLayersCount = 2;
    net.nodesInEachLayer = 10;
    net.bias = 0.0f;
    net.trainingIterations = 50;
    net.learningRate = 0.00001f;
    
    initNeuralNetwork(&net);
    
    for (i32 b = 0; b < 500; ++b)
    {
        train(&net);
        setWeightsToAveraged(&net);
    }
    
    //setWeightsToAveraged(&net);
    
    //printNet(&net);
    
    i32 input = 3;
    feedInput(&net, input);
    i32 output = 2*input;
    propagateForward(&net);
    calculateErrors(&net, output);
    backpropagate(&net);
    
    r32 totalError = getTotalError(&net, output);
    
    printf("Total error after training: %f%%\n", totalError);
    
    return 0;
}
