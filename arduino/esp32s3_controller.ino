#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <Adafruit_NeoPixel.h>

#define SERVICE_UUID         "12345678-1234-1234-1234-123456789abc"
#define CHARACTERISTIC_UUID  "87654321-4321-4321-4321-cba987654321"

#define RGB_PIN 48
#define NUMPIXELS 1

// L298N
#define IN3 10
#define IN4 11

Adafruit_NeoPixel pixel(NUMPIXELS, RGB_PIN, NEO_GRB + NEO_KHZ800);

char currentCommand = '0';

void motorStop()
{
    digitalWrite(IN3, LOW);
    digitalWrite(IN4, LOW);
}

void motorForward()
{
    digitalWrite(IN3, HIGH);
    digitalWrite(IN4, LOW);
}

void motorBackward()
{
    digitalWrite(IN3, LOW);
    digitalWrite(IN4, HIGH);
}

class MyServerCallbacks : public BLEServerCallbacks
{
    void onConnect(BLEServer* pServer)
    {
        Serial.println("Client connected");
    }

    void onDisconnect(BLEServer* pServer)
    {
        Serial.println("Client disconnected");

        motorStop();

        BLEDevice::startAdvertising();
        Serial.println("Advertising restarted");
    }
};

class MyCallbacks : public BLECharacteristicCallbacks
{
    void onWrite(BLECharacteristic *pCharacteristic)
    {
        String value = pCharacteristic->getValue();

        if (value.length() == 0)
            return;

        currentCommand = value[0];

        Serial.print("Received: ");
        Serial.println(currentCommand);

        switch(currentCommand)
        {
            // Command 1: LONG
            // GREEN LED
            // Forward 3 seconds
            case '1':
            {
                pixel.setPixelColor(0, pixel.Color(0, 255, 0));
                pixel.show();

                Serial.println("LONG: Forward 3s");

                motorForward();
                delay(3000);
                motorStop();

                pixel.setPixelColor(0, pixel.Color(0, 0, 0));
                pixel.show();

                break;
            }

            // Command 2: SHORT
            // BLUE LED
            // Forward 1 second
            case '2':
            {
                pixel.setPixelColor(0, pixel.Color(0, 0, 255));
                pixel.show();

                Serial.println("SHORT: Forward 1s");

                motorForward();
                delay(1000);
                motorStop();

                pixel.setPixelColor(0, pixel.Color(0, 0, 0));
                pixel.show();

                break;
            }

            // Command 3: BACK
            // RED LED
            // Backward 3 seconds
            case '3':
            {
                pixel.setPixelColor(0, pixel.Color(255, 0, 0));
                pixel.show();

                Serial.println("BACK: Backward 3s");

                motorBackward();
                delay(3000);
                motorStop();

                pixel.setPixelColor(0, pixel.Color(0, 0, 0));
                pixel.show();

                break;
            }

            // Command 4: EXTRA
            // YELLOW LED
            // Backward 1 second
            case '4':
            {
                pixel.setPixelColor(0, pixel.Color(255, 165, 0));
                pixel.show();

                Serial.println("EXTRA: Backward 1s");

                motorBackward();
                delay(1000);
                motorStop();

                pixel.setPixelColor(0, pixel.Color(0, 0, 0));
                pixel.show();

                break;
            }

            default:
            {
                Serial.println("Unknown command");
                break;
            }
        }
    }
};

BLEServer* pServer;
BLEService* pService;
BLECharacteristic* pCharacteristic;

void setup()
{
    Serial.begin(115200);
    delay(1000);

    // Motor pins
    pinMode(IN3, OUTPUT);
    pinMode(IN4, OUTPUT);
    motorStop();

    // RGB
    pixel.begin();
    pixel.clear();
    pixel.show();

    Serial.println("Starting BLE...");

    BLEDevice::init("ESP32S3_BLE");

    pServer = BLEDevice::createServer();
    pServer->setCallbacks(new MyServerCallbacks());

    pService = pServer->createService(SERVICE_UUID);

    pCharacteristic = pService->createCharacteristic(
        CHARACTERISTIC_UUID,
        BLECharacteristic::PROPERTY_READ |
        BLECharacteristic::PROPERTY_WRITE
    );

    pCharacteristic->setCallbacks(new MyCallbacks());
    pCharacteristic->setValue("Ready");

    pService->start();

    BLEAdvertising* pAdvertising = BLEDevice::getAdvertising();
    pAdvertising->start();

    Serial.println("BLE advertising started");

    // Startup flash: white pulse
    pixel.setPixelColor(0, pixel.Color(255, 255, 255));
    pixel.show();
    delay(200);
    pixel.setPixelColor(0, pixel.Color(0, 0, 0));
    pixel.show();
}

void loop()
{
    delay(100);
}
