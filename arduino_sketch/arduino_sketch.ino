#include <minist-keras-model_inferencing.h>
#include <Arduino_OV767X.h>

#define ORIGINAL_WIDTH 160
#define ORIGINAL_HEIGHT 120
#define TARGET_WIDTH 28
#define TARGET_HEIGHT 28

unsigned short pixels[ORIGINAL_WIDTH * ORIGINAL_HEIGHT]; // QQVGA: 160x120 X 16 bytes per pixel (RGB565)
unsigned short cropped_pixels[ORIGINAL_HEIGHT * ORIGINAL_HEIGHT]; // 120x120
unsigned short resized_pixels[TARGET_WIDTH*TARGET_HEIGHT]; // 28x28
unsigned char greyscale_pixels[TARGET_WIDTH * TARGET_HEIGHT];
float normalized_pixels[TARGET_WIDTH * TARGET_HEIGHT];

void setup() {
  Serial.begin(9600);
  while (!Serial);

  Serial.println("OV767X Camera Capture");
  Serial.println();

  if (!Camera.begin(QQVGA, RGB565, 1)) {
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

  Serial.println("Send the 'c' character to read a frame ...");
  Serial.println();
}

void cropImage(unsigned short *original, unsigned short *cropped) {
  // Calculate the cropping bounds
  int cropStartX = (ORIGINAL_WIDTH - ORIGINAL_HEIGHT) / 2;
  int cropEndX = cropStartX + ORIGINAL_HEIGHT;

  // Crop the center region
  for (int y = 0; y < ORIGINAL_HEIGHT; y++) {
    for (int x = 0; x < ORIGINAL_HEIGHT; x++) {
      int originalIndex = (y * ORIGINAL_WIDTH) + (x + cropStartX);
      int croppedIndex = y * ORIGINAL_HEIGHT + x;
      cropped[croppedIndex] = original[originalIndex];
    }
  }
}

void resizeImage(unsigned short *original, unsigned short *resized) {
  float x_ratio = (float)ORIGINAL_HEIGHT / TARGET_WIDTH;
  float y_ratio = (float)ORIGINAL_HEIGHT / TARGET_HEIGHT;

  for (int y = 0; y < TARGET_HEIGHT; y++) {
    for (int x = 0; x < TARGET_WIDTH; x++) {
      int px = (int)(x * x_ratio);
      int py = (int)(y * y_ratio);

      int originalIndex = (py * ORIGINAL_WIDTH) + px;
      int resizedIndex = (y * TARGET_WIDTH) + x;

      resized[resizedIndex] = original[originalIndex];
    }
  }
}

void rgb565ToGreyscale(const unsigned short *rgb565Image, unsigned char *greyscaleImage) {
  for (int i = 0; i < ORIGINAL_WIDTH * ORIGINAL_HEIGHT; i++) {
    unsigned short pixel = rgb565Image[i];

    // Extract individual color components
    unsigned short red = (pixel >> 11) & 0x1F;
    unsigned short green = (pixel >> 5) & 0x3F;
    unsigned short blue = pixel & 0x1F;

    // Convert to greyscale using luminosity method
    unsigned char greyscale = (unsigned char)(0.299 * red + 0.587 * green + 0.114 * blue);

    greyscaleImage[i] = greyscale;
  }
}

void normalizeImage(unsigned char *greyscaleImage, float *normalizedImage) {
  for (int i = 0; i < TARGET_WIDTH * TARGET_HEIGHT; i++) {
    normalizedImage[i] = greyscaleImage[i] / 255.0f;
  }
}

void displayImageOnSerial(const unsigned char *image, int width, int height) {
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

/**
 * @brief      Convert RGB565 raw camera buffer to RGB888
 *
 * @param[in]   offset       pixel offset of raw buffer
 * @param[in]   length       number of pixels to convert
 * @param[out]  out_buf      pointer to store output image
 */
int ei_camera_cutout_get_data(size_t offset, size_t length, float *out_ptr) {
    size_t pixel_ix = offset * 2; 
    size_t bytes_left = length;
    size_t out_ptr_ix = 0;

    // read byte for byte
    while (bytes_left != 0) {
        // grab the value and convert to r/g/b
        uint16_t pixel = (resized_pixels[pixel_ix] << 8) | resized_pixels[pixel_ix+1];
        uint8_t r, g, b;
        r = ((pixel >> 11) & 0x1f) << 3;
        g = ((pixel >> 5) & 0x3f) << 2;
        b = (pixel & 0x1f) << 3;

        // then convert to out_ptr format
        float pixel_f = (r << 16) + (g << 8) + b;
        ei_printf("%.5f",pixel_f);  // Print with four decimal places
        ei_printf(" ");
        out_ptr[out_ptr_ix] = pixel_f;

        // and go to the next pixel
        out_ptr_ix++;
        pixel_ix+=2;
        bytes_left--;
    }

    // and done!
    return 0;
}

void predict(float *normalizedImage) {
  // summary of inferencing settings (from model_metadata.h)
  ei_printf("Inferencing settings:\n");
  ei_printf("\tImage resolution: %dx%d\n", EI_CLASSIFIER_INPUT_WIDTH, EI_CLASSIFIER_INPUT_HEIGHT);
  ei_printf("\tFrame size: %d\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
  ei_printf("\tNo. of classes: %d\n", sizeof(ei_classifier_inferencing_categories) / sizeof(ei_classifier_inferencing_categories[0]));

  ei::signal_t signal;
  signal.total_length = EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT;
  signal.get_data = &ei_camera_cutout_get_data;

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
    cropImage(pixels, cropped_pixels);
    resizeImage(cropped_pixels, resized_pixels);
    rgb565ToGreyscale(resized_pixels, greyscale_pixels);
    normalizeImage(greyscale_pixels, normalized_pixels);

    int numPixels = TARGET_WIDTH * TARGET_HEIGHT;

    for (int i = 0; i < numPixels; i++) {
      float p = normalized_pixels[i];

      //Serial.print(p, 4);  // Print with four decimal places
      //Serial.print(' ');
    }
    displayImageOnSerial(greyscale_pixels, TARGET_WIDTH, TARGET_HEIGHT);
    predict(normalized_pixels);
  }
}
