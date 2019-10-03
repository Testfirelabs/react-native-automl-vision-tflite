#import "AutomlVisionTflite.h"
#import "TFLTensorFlowLite.h"
#import <stdlib.h>

@interface LoadedModel : NSObject
{
    TFLInterpreter *interpreter;
    NSArray<NSString *> *labels;
}
@property (retain) TFLInterpreter *interpreter;
@property (retain) NSArray<NSString *> *labels;
@end

@implementation LoadedModel
@synthesize interpreter, labels;

- (instancetype)initWithModelPath:(NSString *)modelPath labelsPath:(NSString *)labelsPath {
    // TODO: return nil unless both modelPath and labelsPath aren't nil
    if ((self = [super init])) {
        NSError *err = nil;

        // load model interpreter
        interpreter = [[TFLInterpreter alloc] initWithModelPath:modelPath error:&err];
        if (interpreter) {
            // success
        } else if (err) {
            NSLog(@"Failed to load model: %@", [err localizedDescription]);
        }

        // load labels
        NSString *labelsContents = [NSString stringWithContentsOfFile:labelsPath encoding:NSUTF8StringEncoding error:&err];
        if (err) {
            NSLog(@"Failed to load labels: %@", [err localizedDescription]);
        } else {
            labels = [labelsContents componentsSeparatedByString:@"\n"];
        }
    }
    return self;
}
@end

@interface AutomlVisionTflite()
+ (NSMutableDictionary*)loadedModels;
@end

@implementation AutomlVisionTflite

RCT_EXPORT_MODULE();

+ (NSMutableDictionary*)loadedModels {
    static NSMutableDictionary *_loadedModels = nil;

    // might also be achievable with
    // @synchronized(self) { if (sharedMyManager == nil) ... }
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        _loadedModels = [NSMutableDictionary dictionary];
    });

    return _loadedModels;
}

RCT_EXPORT_METHOD(loadModel:(NSString *)modelFile
                  withLabels:(NSString *)labelsFile
                  callback:(RCTResponseSenderBlock)callback)
{
    NSString* modelPath = [[NSBundle mainBundle] pathForResource:modelFile ofType:nil];
    NSString* labelsPath = [[NSBundle mainBundle] pathForResource:labelsFile ofType:nil];

    LoadedModel* model = [[LoadedModel alloc] initWithModelPath:modelPath labelsPath:labelsPath];
    if (model) {
        // loaded successfully!
        
        // create random id and save for later
        NSString *modelId = [NSString stringWithFormat:@"%010d%010d", arc4random_uniform(0xFFFFFFFF), arc4random_uniform(0xFFFFFFFF)];
        [[AutomlVisionTflite loadedModels] setObject:model forKey:modelId];
        
        // return id in callback
        callback(@[[NSNull null], modelId]);
    } else {
        callback(@[@"Failed to load model", [NSNull null]]);
    }
}

RCT_EXPORT_METHOD(runModelOnImage:(NSString *)modelId
                  imagePath:(NSString*)imagePath
                  numResults:(int)numResults
                  threshold:(float)threshold
                  callback:(RCTResponseSenderBlock)callback) {
    
//    // TODO: replace with lookup into NSDictionary holding models
//    if (!interpreter) {
//        NSLog(@"Failed to construct interpreter.");
//        callback(@[@"Failed to construct interpreter."]);
//    }
//
//    image_path = [image_path stringByReplacingOccurrencesOfString:@"file://" withString:@""];
//    int input_size;
//    feedInputTensorImage(image_path, input_mean, input_std, &input_size);
//
//    if (interpreter->Invoke() != kTfLiteOk) {
//        NSLog(@"Failed to invoke!");
//        callback(@[@"Failed to invoke!"]);
//    }
//
//    float* output = interpreter->typed_output_tensor<float>(0);
//
//    if (output == NULL)
//        callback(@[@"No output!"]);
//
//    const unsigned long output_size = labels.size();
//    NSMutableArray* results = GetTopN(output, output_size, num_results, threshold);
//
    NSArray* results = @[];
    callback(@[[NSNull null], results]);
}

RCT_EXPORT_METHOD(close:(NSString *)modelId)
{
    [[AutomlVisionTflite loadedModels] removeObjectForKey:modelId];
}

@end
