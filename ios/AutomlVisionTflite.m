#import "AutomlVisionTflite.h"
#import "TFLTensorFlowLite.h"

#import <CoreGraphics/CoreGraphics.h>
#import <stdlib.h>

@interface LoadedModel : NSObject
{
    TFLInterpreter *interpreter;
    NSArray<NSString *> *labels;
    TFLTensor *imageInput;
    uint inputWidth;
    uint inputHeight;
    TFLTensor *resultsOutput;
}
@property (retain) TFLInterpreter *interpreter;
@property (retain) NSArray<NSString *> *labels;
@property (retain) TFLTensor *imageInput;
@property uint inputWidth;
@property uint inputHeight;
@property (retain) TFLTensor *resultsOutput;
@end

@implementation LoadedModel
@synthesize interpreter, labels, imageInput, inputWidth, inputHeight, resultsOutput;

- (instancetype)initWithModelPath:(NSString *)modelPath labelsPath:(NSString *)labelsPath {
    // TODO: return nil unless both modelPath and labelsPath aren't nil
    if ((self = [super init])) {
        NSError *err = nil;

        // load model interpreter
        interpreter = [[TFLInterpreter alloc] initWithModelPath:modelPath error:&err];
        if (err) {
            NSLog(@"Failed to load model: %@", [err localizedDescription]);
            return nil;
        } else if (interpreter) {
            NSLog(@"Successfully loaded model");
        } else {
            NSLog(@"Failed to load model: unknown error");
            return nil;
        }
        
        [interpreter allocateTensorsWithError:&err];
        if (err) {
            NSLog(@"Failed to allocate tensors: %@", [err localizedDescription]);
            return nil;
        }
        
        imageInput = [interpreter inputTensorAtIndex:0 error:&err];
        if (err) {
            NSLog(@"Failed to load input tensor: %@", [err localizedDescription]);
            return nil;
        }
        NSArray<NSNumber *> *inputSize = [imageInput shapeWithError:&err];
        if (err) {
            NSLog(@"Failed to get input tensor size: %@", [err localizedDescription]);
            return nil;
        }
        NSLog(@"Input size is: %@", inputSize);
        inputWidth = [inputSize[1] unsignedIntValue];
        inputHeight = [inputSize[2] unsignedIntValue];
        
        resultsOutput = [interpreter outputTensorAtIndex:0 error:&err];
        if (err) {
            NSLog(@"Failed to load output tensor: %@", [err localizedDescription]);
            return nil;
        }
        
        // load labels
        NSString *labelsContents = [NSString stringWithContentsOfFile:labelsPath encoding:NSUTF8StringEncoding error:&err];
        if (err) {
            NSLog(@"Failed to load labels: %@", [err localizedDescription]);
            return nil;
        } else {
            labels = [labelsContents componentsSeparatedByCharactersInSet:[NSCharacterSet newlineCharacterSet]];
            // TODO: the last line might be empty; if so, remove last line
            NSLog(@"Loaded labels: %@", labels);
        }
    }
    return self;
}

// runImage
- (NSArray *)runImage:(NSData *)imageBytes numberOfResults:(int)numResults threshold:(float)threshold error:(NSError **)error
{
    // TODO: more image resizing here; that way we don't have to expose the input width/height
    
    NSError *err = nil;
    
    // load input tensor
    [imageInput copyData:imageBytes error:&err];
    // check if err
    
    // run model
    [interpreter invokeWithError:&err];
    // check if err
    
    // read results from output tensor
    NSData *outputData = [resultsOutput dataWithError:&err];
    // it might not be outputData long, but it can't be longer than outputData
    NSMutableArray<NSDictionary *> *results = [NSMutableArray arrayWithCapacity:[outputData length]];
    
    // while it would normally be preferrable to enumerate the bytes so they don't get flattened
    // that doesn't matter in this case
    uint8_t *outputBytes = (uint8_t *)[outputData bytes];
    for (uint i = 0; i < [outputData length]; ++i) {
        float confidence = (float)((outputBytes[i] & 0xff) / 255.0f);
        if (confidence >= threshold) {
            // get label and add to array
            NSString *label = i < [labels count] ? labels[i] : @"unknown";
            [results addObject:@{@"confidence": [NSNumber numberWithFloat:confidence], @"label": label}];
        }
    }

    // sort descending and return only the min(count, numResults)
    NSArray *sortedArray = [results sortedArrayUsingComparator:^NSComparisonResult(id a, id b) {
        NSNumber *first = [(NSDictionary *)a objectForKey:@"confidence"];
        NSNumber *second = [(NSDictionary *)b objectForKey:@"confidence"];
        return [second compare:first];
    }];

    if ([sortedArray count] <= numResults) {
        return sortedArray;
    } else {
        NSRange subset;
        subset.location = 0;
        subset.length = numResults;
        return [sortedArray subarrayWithRange:subset];
    }
}
@end

@interface AutomlVisionTflite()
+ (NSMutableDictionary *)loadedModels;
+ (NSData *)loadImage:(NSString *)imagePath width:(uint)width height:(uint)height error:(NSError **)error;
@end

@implementation AutomlVisionTflite

RCT_EXPORT_MODULE();

+ (NSMutableDictionary *)loadedModels {
    static NSMutableDictionary *_loadedModels = nil;

    // might also be achievable with
    // @synchronized(self) { if (sharedMyManager == nil) ... }
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        _loadedModels = [NSMutableDictionary dictionary];
    });

    return _loadedModels;
}

+ (NSData *)loadImage:(NSString *)imagePath width:(uint)width height:(uint)height error:(NSError *__autoreleasing *)error
{
    NSURL *imageUrl = [NSURL fileURLWithPath:imagePath];
    
    CGImageSourceRef imgSrc = CGImageSourceCreateWithURL((CFURLRef)imageUrl, NULL);
    CGImageRef originalImg = CGImageSourceCreateImageAtIndex(imgSrc, 0, NULL);
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = CGBitmapContextCreate(NULL,
                                                 width,
                                                 height,
                                                 8,
                                                 (width * 4),
                                                 colorSpace,
                                                 kCGImageAlphaNoneSkipFirst);

    CGContextDrawImage(context, CGContextGetClipBoundingBox(context), originalImg);

    NSMutableData *scaledImage = [NSMutableData dataWithLength:(width * height * 3)];
    uint8_t *scaledRgbPixels = [scaledImage mutableBytes];
    uint8_t *scaledArgbPixels = (uint8_t *)CGBitmapContextGetData(context);
    
    uint scaledRgbOffset = 0;
    uint scalledArgbOffset = 1;
    for (uint y = 0; y < height; ++y) {
        for (uint x = 0; x < width; ++x, scalledArgbOffset++) {
            scaledRgbPixels[scaledRgbOffset++] = scaledArgbPixels[scalledArgbOffset++];
            scaledRgbPixels[scaledRgbOffset++] = scaledArgbPixels[scalledArgbOffset++];
            scaledRgbPixels[scaledRgbOffset++] = scaledArgbPixels[scalledArgbOffset++];
        }
    }
    
    // free(scaledArgbPixels);
    CGContextRelease(context);
    CGColorSpaceRelease(colorSpace);
    CFRelease(originalImg);
    CFRelease(imgSrc);
    
    return scaledImage;
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
    
    LoadedModel *model = [[AutomlVisionTflite loadedModels] objectForKey:modelId];
    // check if nil
    
    imagePath = [imagePath stringByReplacingOccurrencesOfString:@"file://" withString:@""];

    NSError *err;
    NSData *imageBytes = [AutomlVisionTflite loadImage:imagePath width:[model inputWidth] height:[model inputHeight] error:&err];
    // check if err
    
    NSArray *results = [model runImage:imageBytes numberOfResults:numResults threshold:threshold error:&err];
    // check if err
    
    callback(@[[NSNull null], results]);
}

RCT_EXPORT_METHOD(close:(NSString *)modelId)
{
    [[AutomlVisionTflite loadedModels] removeObjectForKey:modelId];
}

@end

