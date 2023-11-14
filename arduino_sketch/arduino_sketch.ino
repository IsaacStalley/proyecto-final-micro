#include <minist-keras-model_inferencing.h>
#include <Arduino_OV767X.h>

#define ORIGINAL_WIDTH 160
#define ORIGINAL_HEIGHT 120
#define TARGET_WIDTH 28
#define TARGET_HEIGHT 28

unsigned short pixels[ORIGINAL_WIDTH * ORIGINAL_HEIGHT]; // QQVGA: 160x120 X 16 bytes per pixel (RGB565)
unsigned short resized_pixels[TARGET_WIDTH*TARGET_HEIGHT];
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

void resizeImage(unsigned short *original, unsigned short *resized) {
  float x_ratio = (float)ORIGINAL_WIDTH / TARGET_WIDTH;
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

void loop() {
  if (Serial.read() == 'c') {
    Serial.println("Reading frame");
    Serial.println();
    Camera.readFrame(pixels);
    resizeImage(pixels, resized_pixels);
    rgb565ToGreyscale(resized_pixels, greyscale_pixels);
    normalizeImage(greyscale_pixels, normalized_pixels);

    int numPixels = TARGET_WIDTH * TARGET_HEIGHT;

    for (int i = 0; i < numPixels; i++) {
      float p = normalized_pixels[i];

      Serial.print(p);
      Serial.print('/');
    }
  }
}
