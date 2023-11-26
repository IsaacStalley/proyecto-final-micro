#include <minist-keras-model_inferencing.h>
#include <Arduino_OV767X.h>
#include <SPI.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// Constants for OLED display
#define SCREEN_WIDTH 128 // OLED display width, in pixels
#define SCREEN_HEIGHT 64 // OLED display height, in pixels
#define OLED_RESET -1    // Reset pin # (or -1 if sharing Arduino reset pin)
#define SCREEN_ADDRESS 0x3C

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// Constants for camera and image processing
#define ORIGINAL_WIDTH 176
#define ORIGINAL_HEIGHT 144
#define TARGET_WIDTH 28
#define TARGET_HEIGHT 28
#define UPPER_CONTRAST 150
#define LOWER_CONTRAST 100

// Image arrays
uint16_t pixels[ORIGINAL_WIDTH * ORIGINAL_HEIGHT];
uint8_t greyscale_pixels[ORIGINAL_WIDTH * ORIGINAL_HEIGHT];
uint8_t resized_pixels[TARGET_WIDTH * TARGET_HEIGHT];
float normalized_pixels[TARGET_WIDTH * TARGET_HEIGHT];

// Functions
void setup();
void displayImage(uint8_t* imageArray, uint8_t width, uint8_t height);
void displayNumber(uint8_t number);
void displayPercentage(uint8_t percentage);
void displayBattery(float battery);
float getBatteryLevel();
void scaleAndCenter(uint8_t* inputNumber, uint8_t* outputNumber, uint8_t boundingBoxWidth, uint8_t boundingBoxHeight);
void cropAndResize(uint8_t* inputImage, uint8_t* outputImage);
void contrastStretching(uint8_t* img);
void byteReverse(uint16_t* rgb565Image);
void rgb565ToGreyscale(const uint16_t* rgb565Image, uint8_t* greyscaleImage);
void normalizeImage(uint8_t* greyscaleImage, float* normalizedImage);
void displayImageOnSerial(uint8_t* image, uint8_t width, uint8_t height);
void printImageStrings();
int get_prediction_data(size_t offset, size_t length, float* out_ptr);
void predict();
void loop();

// Setup function
void setup() {
  Serial.begin(9600);
  while (!Serial);

  // Initialize OLED display
  display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS);

  // Show initial display buffer contents on the screen
  display.display();
  delay(1000);

  // Clear the buffer
  display.clearDisplay();

  // Initialize camera
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
}

// Function to display an image on OLED
void displayImage(uint8_t* imageArray, uint8_t width, uint8_t height) {
  // Calculate the starting position to center the scaled image
  display.clearDisplay();
  uint8_t startX = (SCREEN_WIDTH - TARGET_WIDTH) / 2;

  for (uint8_t y = 0; y < height; y++) {
    for (uint8_t x = 0; x < width; x++) {
      int index = y * width + x;
      if (imageArray[index] > 0) {
        display.drawPixel(x + startX, y, SSD1306_WHITE);
      }
    }
  }
  display.display();
}

// Function to display a number on OLED
void displayNumber(uint8_t number) {
  display.clearDisplay();
  display.setTextSize(3);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(10, 5);
  display.cp437(true);
  display.print("Pred:");
  display.print(number);
  display.display();
}

// Function to display a percentage on OLED
void displayPercentage(uint8_t percentage) {
  display.setTextSize(2);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(85, 50);
  display.print(percentage);
  display.print("%");
  display.display();
}

// Function to display battery voltage on OLED
void displayBattery(float battery) {
  display.setTextSize(2);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(5, 50);
  display.print(battery);
  display.print("V");
  display.display();
}

// Function to get battery voltage
float getBatteryLevel() {
  int sensorValue = analogRead(A6);
  float batValue = sensorValue * 3 * 3.3f / 1024.0f;
  return batValue;
}

// Function to scale and center an image
void scaleAndCenter(uint8_t* inputNumber, uint8_t* outputNumber, uint8_t boundingBoxWidth, uint8_t boundingBoxHeight) {
  // Constants for the target dimensions
  const uint8_t targetHeight = 20;
  float xy_ratio = (float)boundingBoxWidth / boundingBoxHeight;
  const uint8_t targetWidth = targetHeight * xy_ratio + targetHeight * 0.2f;

  float y_ratio = (float)boundingBoxHeight / targetHeight;
  float x_ratio = (float)boundingBoxWidth / targetWidth;

  // Calculate the starting position to center the scaled image
  uint8_t startX = (TARGET_WIDTH - targetWidth) / 2;
  uint8_t startY = (TARGET_HEIGHT - targetHeight) / 2;

  // Initialize output array with zeros
  for (int i = 0; i < TARGET_WIDTH * TARGET_HEIGHT; ++i) {
    outputNumber[i] = 0;
  }

  for (uint8_t y = 0; y < targetHeight; y++) {
    for (uint8_t x = 0; x < targetWidth; x++) {
      uint8_t px = (uint8_t)(x * x_ratio);
      uint8_t py = (uint8_t)(y * y_ratio);

      int originalIndex = (py * boundingBoxWidth) + px;
      int resizedIndex = (startY + y) * TARGET_WIDTH + (startX + x);

      outputNumber[resizedIndex] = inputNumber[originalIndex];
    }
  }
}

// Function to crop and resize an image
void cropAndResize(uint8_t* inputImage, uint8_t* outputImage) {
  // Find bounding box around the number
  uint8_t minX = ORIGINAL_WIDTH;
  uint8_t minY = ORIGINAL_HEIGHT;
  uint8_t maxX = 0;
  uint8_t maxY = 0;

  for (uint8_t y = 0; y < ORIGINAL_HEIGHT; y++) {
    for (uint8_t x = 0; x < ORIGINAL_WIDTH; x++) {
      int index = y * ORIGINAL_WIDTH + x;
      if (inputImage[index] >= UPPER_CONTRAST) {
        // Update bounding box
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  minX = (minX > 0) ? minX - 1 : minX;
  minY = (minY > 0) ? minY - 1 : minY;
  maxX = (maxX < ORIGINAL_WIDTH) ? maxX + 1 : maxX;
  maxY = (maxY < ORIGINAL_HEIGHT) ? maxY + 1 : maxY;

  // Check if bounding box is greater than 28 in either height or width
  uint8_t boundingBoxWidth = maxX - minX + 1;
  uint8_t boundingBoxHeight = maxY - minY + 1;
  uint8_t number[boundingBoxWidth * boundingBoxHeight];

  for (int i = 0; i < boundingBoxWidth * boundingBoxHeight; ++i) {
    number[i] = 0;
  }

  // Crop the image based on the resized bounding box
  for (uint8_t y = minY; y <= minY + boundingBoxHeight - 1; y++) {
    for (uint8_t x = minX; x <= minX + boundingBoxWidth - 1; x++) {
      int inputIndex = y * ORIGINAL_WIDTH + x;
      int outputIndex = (y - minY) * boundingBoxWidth + (x - minX);

      // Check if the current pixel value is greater than 250
      if (inputImage[inputIndex] >= UPPER_CONTRAST) {
        number[outputIndex] = inputImage[inputIndex];
      } else if (inputImage[(y - 1) * ORIGINAL_WIDTH + x] >= UPPER_CONTRAST) {
        number[outputIndex] = inputImage[inputIndex];
      } else if (inputImage[(y + 1) * ORIGINAL_WIDTH + x] >= UPPER_CONTRAST) {
        number[outputIndex] = inputImage[inputIndex];
      } else if (inputImage[y * ORIGINAL_WIDTH + (x - 1)] >= UPPER_CONTRAST) {
        number[outputIndex] = inputImage[inputIndex];
      } else if (inputImage[y * ORIGINAL_WIDTH + (x + 1)] >= UPPER_CONTRAST) {
        number[outputIndex] = inputImage[inputIndex];
      }
    }
  }
  scaleAndCenter(number, outputImage, boundingBoxWidth, boundingBoxHeight);
}

// Function to perform contrast stretching on an image
void contrastStretching(uint8_t* img) {
  // Find the minimum and maximum pixel values
  uint8_t min_val = 255;
  uint8_t max_val = 0;

  for (int i = 0; i < ORIGINAL_WIDTH * ORIGINAL_HEIGHT; i++) {
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
  for (int i = 0; i < ORIGINAL_WIDTH * ORIGINAL_HEIGHT; i++) {
    img[i] = ((img[i] - min_val) * 255) / (max_val - min_val);
  }
}

// Function to byte-reverse RGB565 image
void byteReverse(uint16_t* rgb565Image) {
  uint16_t pixel;
  for (int i = 0; i < ORIGINAL_WIDTH * ORIGINAL_HEIGHT; i++) {
    pixel = ((rgb565Image[i] & 0xFF) << 8) | (rgb565Image[i] >> 8);
    rgb565Image[i] = pixel;
  }
}

// Function to convert RGB565 image to greyscale
void rgb565ToGreyscale(const uint16_t* rgb565Image, uint8_t* greyscaleImage) {
  for (int i = 0; i < ORIGINAL_WIDTH * ORIGINAL_HEIGHT; i++) {
    uint16_t pixel = rgb565Image[i];

    // Extract red, green, and blue components
    uint8_t red = ((pixel >> 11) & 0x1f) << 3;
    uint8_t green = ((pixel >> 5) & 0x3f) << 2;
    uint8_t blue = (pixel & 0x1f) << 3;
    uint8_t grayscale = ((red << 16) + (green << 8) + blue) * 255;

    greyscaleImage[i] = grayscale;
  }
}

// Function to normalize greyscale image
void normalizeImage(uint8_t* greyscaleImage, float* normalizedImage) {
  for (int i = 0; i < TARGET_WIDTH * TARGET_HEIGHT; i++) {
    normalizedImage[i] = greyscaleImage[i] / 255.0f;
  }
}

// Function to display image on serial monitor
void displayImageOnSerial(uint8_t* image, uint8_t width, uint8_t height) {
  for (uint8_t y = 0; y < height; y++) {
    for (uint8_t x = 0; x < width; x++) {
      // Print a character based on pixel intensity
      char pixelChar = (image[y * width + x] > 128) ? '#' : ' ';
      Serial.print(pixelChar);
    }
    Serial.println(); // Move to the next line for the next row
  }
}

// Strings to visualize images in the python script
void printImageStrings() {
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
}

// Function to get prediction data for inferencing
int get_prediction_data(size_t offset, size_t length, float* out_ptr) {
  for (int i = 0; i < TARGET_WIDTH * TARGET_HEIGHT; i++) {
    out_ptr[i] = normalized_pixels[i];
  }
  return 0;
}

// Function to perform prediction using the model
void predict() {
  // Create signal for inferencing
  ei::signal_t signal;
  signal.total_length = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
  signal.get_data = &get_prediction_data;

  // Run the impulse: DSP, neural network, and the Anomaly algorithm
  ei_impulse_result_t result = {0};
  EI_IMPULSE_ERROR ei_error = run_classifier(&signal, &result, false);

  // Print the predictions
  ei_printf("Predictions (DSP: %d ms., Classification: %d ms., Anomaly: %d ms.): \n",
            result.timing.dsp, result.timing.classification, result.timing.anomaly);
  uint8_t largest_number;
  float largest_value = 0;
  for (uint8_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
    ei_printf("    %s: %.5f\n", result.classification[ix].label,
              result.classification[ix].value);
    if (result.classification[ix].value > largest_value) {
      largest_value = result.classification[ix].value;
      largest_number = ix;
    }
  }
  uint8_t percent = largest_value * 100;
  displayNumber(largest_number);
  displayPercentage(percent);
}

// Main loop
void loop() {
  Serial.println("Reading frame");
  Serial.println();
  Camera.readFrame(pixels);
  byteReverse(pixels);
  rgb565ToGreyscale(pixels, greyscale_pixels);
  contrastStretching(greyscale_pixels);
  cropAndResize(greyscale_pixels, resized_pixels);
  normalizeImage(resized_pixels, normalized_pixels);
  printImageStrings();
  //displayImageOnSerial(resized_pixels, TARGET_WIDTH, TARGET_HEIGHT);
  predict();
  displayBattery(getBatteryLevel());
  delay(2000);
}