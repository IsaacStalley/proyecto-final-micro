
#include <minist-keras-model_inferencing.h>
#include <Arduino_OV767X.h>

#define ORIGINAL_WIDTH 176
#define ORIGINAL_HEIGHT 144
#define TARGET_WIDTH 28
#define TARGET_HEIGHT 28
#define UPPER_CONTRAST 128

uint16_t pixels[ORIGINAL_WIDTH * ORIGINAL_HEIGHT]; // QQVGA: 160x120 X 2 bytes per pixel (RGB565)
uint8_t greyscale_pixels[ORIGINAL_WIDTH * ORIGINAL_HEIGHT];
uint8_t resized_pixels[TARGET_WIDTH*TARGET_HEIGHT]; // 28x28
float normalized_pixels[TARGET_WIDTH * TARGET_HEIGHT];
int bytesPerFrame;

void setup() {
  Serial.begin(9600);
  while (!Serial);

  Serial.println("OV767X Camera Capture");
  Serial.println();

  if (!Camera.begin(QCIF, RGB565, 1)) {
    Serial.println("Failed to initialize camera!");
    while (1);
  }

  Serial.println("Camera settings:");
  Serial.print("\twidth = ");
  Serial.println(Camera.width());
  Serial.print("\theight = ");
  Serial.println(Camera.height());
  Serial.print("\tbits per pixel = ");
  Serial.println(Camera.bitsPerPixel());
  Serial.println();
  bytesPerFrame = Camera.width() * Camera.height() * Camera.bytesPerPixel();
  Serial.println("Send the 'c' character to read a frame ...");
  Serial.println();
}

void scaleAndCenter(uint8_t* inputNumber, uint8_t* outputNumber, int boundingBoxWidth, int boundingBoxHeight) {
  // Constants for the target dimensions
  const int targetHeight = 20;
  float xy_ratio = (float)boundingBoxWidth / boundingBoxHeight;
  const int targetWidth = targetHeight*xy_ratio + targetHeight*0.2f;

  float y_ratio = (float)boundingBoxHeight / targetHeight;
  float x_ratio = (float)boundingBoxWidth / targetWidth;

  // Calculate the starting position to center the scaled image
  int startX = (TARGET_WIDTH - targetWidth) / 2;
  int startY = (TARGET_HEIGHT - targetHeight) / 2;

  // Initialize output array with zeros
  for (int i = 0; i < TARGET_WIDTH * TARGET_HEIGHT; ++i) {
      outputNumber[i] = 0;
  }

  for (int y = 0; y < targetHeight; y++) {
    for (int x = 0; x < targetWidth; x++) {
      int px = (int)(x * x_ratio);
      int py = (int)(y * y_ratio);

      int originalIndex = (py * boundingBoxWidth) + px;
      int resizedIndex = (startY + y) * TARGET_WIDTH + (startX + x);

      outputNumber[resizedIndex] = inputNumber[originalIndex];
    }
  }
}

void cropAndResize(uint8_t *inputImage, uint8_t *outputImage) {
    // Find bounding box around the number
    int minX = ORIGINAL_WIDTH;
    int minY = ORIGINAL_HEIGHT;
    int maxX = 0;
    int maxY = 0;

    for (int y = 0; y < ORIGINAL_HEIGHT; y++) {
        for (int x = 0; x < ORIGINAL_WIDTH; x++) {
            int index = y * ORIGINAL_WIDTH + x;
            if (inputImage[index] >= UPPER_CONTRAST) {  // Assuming white represents the number
                // Update bounding box
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }
        }
    }
    minX = (minX > 0) ? minX-1 : minX;
    minY = (minY > 0) ? minY-1 : minY;
    maxX = (maxX < ORIGINAL_WIDTH) ? maxX+1 : maxX;
    maxY = (maxY < ORIGINAL_HEIGHT) ? maxY+1 : maxY;

    // Check if bounding box is greater than 28 in either height or width
    int boundingBoxWidth = maxX - minX + 1;
    int boundingBoxHeight = maxY - minY + 1;
    uint8_t number[boundingBoxWidth*boundingBoxHeight];

    for (int i = 0; i < boundingBoxWidth * boundingBoxHeight; ++i) {
      number[i] = 0;
    }

    // Crop the image based on the resized bounding box
    for (int y = minY; y <= minY + boundingBoxHeight - 1; y++) {
      for (int x = minX; x <= minX + boundingBoxWidth - 1; x++) {
        int inputIndex = y * ORIGINAL_WIDTH + x;
        int outputIndex = (y - minY) * boundingBoxWidth + (x - minX);

        // Check if the current pixel value is greater than 250
        if (inputImage[inputIndex] >= UPPER_CONTRAST) {
          number[outputIndex] = inputImage[inputIndex];
        }
        else  if (inputImage[(y-1) * ORIGINAL_WIDTH + x] >= UPPER_CONTRAST) {
          number[outputIndex] = inputImage[inputIndex];
        }
        else  if (inputImage[(y+1) * ORIGINAL_WIDTH + x] >= UPPER_CONTRAST) {
          number[outputIndex] = inputImage[inputIndex];
        }
        else  if (inputImage[y * ORIGINAL_WIDTH + (x-1)] >= UPPER_CONTRAST) {
          number[outputIndex] = inputImage[inputIndex];
        }
        else  if (inputImage[y * ORIGINAL_WIDTH + (x+1)] >= UPPER_CONTRAST) {
          number[outputIndex] = inputImage[inputIndex];
        }
      }
    }
    scaleAndCenter(number, outputImage, boundingBoxWidth, boundingBoxHeight);
}

void contrastStretching(uint8_t *img) {
    // Find the minimum and maximum pixel values
    uint8_t min_val = 255;
    uint8_t max_val = 0;

    for (int i = 0; i < ORIGINAL_WIDTH*ORIGINAL_HEIGHT; i++) {
        if (img[i] < min_val) {
            min_val = img[i];
        }
        if (img[i] > max_val) {
            max_val = img[i];
        }
    }
    // Check for division by zero
    if (max_val == min_val) {
        // Avoid division by zero by returning without modification
        return;
    }

    // Apply contrast stretching
    for (int i = 0; i < ORIGINAL_WIDTH*ORIGINAL_HEIGHT; i++) {
        img[i] = ((img[i] - min_val) * 255) / (max_val - min_val);
    }
}

void byteReverse(uint16_t *rgb565Image) {
  uint16_t pixel;
  for (int i = 0; i < ORIGINAL_WIDTH * ORIGINAL_HEIGHT; i++) {

    pixel = ((rgb565Image[i]&0xFF)<<8)|(rgb565Image[i]>>8);
    rgb565Image[i] = pixel;
  }
}

void rgb565ToGreyscale(const uint16_t *rgb565Image, uint8_t *greyscaleImage) {
  for (int i = 0; i < ORIGINAL_WIDTH * ORIGINAL_HEIGHT; i++) {
    uint16_t pixel = rgb565Image[i];

    // Extract red, green, and blue components
    uint8_t red = ((pixel >> 11) & 0x1f) << 3;
    uint8_t green = ((pixel >> 5) & 0x3f) << 2;
    uint8_t blue = (pixel & 0x1f) << 3;
    uint8_t grayscale = ((red << 16) + (green << 8) + blue)*255;

    greyscaleImage[i] = grayscale;
  }
}

void normalizeImage(uint8_t *greyscaleImage, float *normalizedImage) {
  for (int i = 0; i < TARGET_WIDTH * TARGET_HEIGHT; i++) {
    normalizedImage[i] = greyscaleImage[i] / 255.0f;
  }
}

void displayImageOnSerial(uint8_t *image, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Print a character based on pixel intensity
            char pixelChar = (image[y * width + x] > 128) ? '#' : ' ';
            Serial.print(pixelChar);
        }
        Serial.println();  // Move to the next line for the next row
    }
}

int get_prediction_data(size_t offset, size_t length, float *out_ptr) {
  for (int i = 0; i < TARGET_WIDTH * TARGET_HEIGHT; i++) {
    out_ptr[i] = normalized_pixels[i];
  }
  return 0;
}

void predict() {
  // summary of inferencing settings (from model_metadata.h)
  ei_printf("Inferencing settings:\n");
  ei_printf("\tImage resolution: %dx%d\n", EI_CLASSIFIER_INPUT_WIDTH, EI_CLASSIFIER_INPUT_HEIGHT);
  ei_printf("\tFrame size: %d\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
  ei_printf("\tNo. of classes: %d\n", sizeof(ei_classifier_inferencing_categories) / sizeof(ei_classifier_inferencing_categories[0]));

  ei::signal_t signal;
  signal.total_length = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
  signal.get_data = &get_prediction_data;

  // run the impulse: DSP, neural network and the Anomaly algorithm
  ei_impulse_result_t result = { 0 };

  EI_IMPULSE_ERROR ei_error = run_classifier(&signal, &result, false);
  if (ei_error != EI_IMPULSE_OK) {
      ei_printf("Failed to run impulse (%d)\n", ei_error);
  }

  // print the predictions
  ei_printf("Predictions (DSP: %d ms., Classification: %d ms., Anomaly: %d ms.): \n",
            result.timing.dsp, result.timing.classification, result.timing.anomaly);

  for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
      ei_printf("    %s: %.5f\n", result.classification[ix].label,
                                  result.classification[ix].value);
  }
}

void loop() {
  if (Serial.read() == 'c') {
    Serial.println("Reading frame");
    Serial.println();
    Camera.readFrame(pixels);
    byteReverse(pixels);
    rgb565ToGreyscale(pixels, greyscale_pixels);
    contrastStretching(greyscale_pixels);
    cropAndResize(greyscale_pixels, resized_pixels);
    normalizeImage(resized_pixels, normalized_pixels);

    int numPixels = ORIGINAL_WIDTH * ORIGINAL_HEIGHT;
    for (int i = 0; i < numPixels; i++) {
      uint16_t p = pixels[i];
      Serial.print("0x");
      if (p < 0x1000) {
        Serial.print('0');
      }

      if (p < 0x0100) {
        Serial.print('0');
      }

      if (p < 0x0010) {
        Serial.print('0');
      }

      Serial.print(p, HEX);  // Print with four decimal places
      Serial.print(", ");
    }
    Serial.println();
    int numPixels2 = TARGET_WIDTH * TARGET_HEIGHT;
    for (int j = 0; j < numPixels2; j++) {
      float f = normalized_pixels[j];
      
      Serial.print(f, 4);  // Print with four decimal places
      Serial.print(", ");
    }
    Serial.println();
    displayImageOnSerial(resized_pixels, TARGET_WIDTH, TARGET_HEIGHT);
    predict();
    
  }
}
