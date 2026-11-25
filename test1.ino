
// GestureRobot - USB Serial Motor Controller

// L298N pins
#define ENA 25
#define IN1 26
#define IN2 27

#define ENB 19
#define IN3 32
#define IN4 33

// PWM settings for ESP32 Core 3.x
const int PWM_FREQ = 1000;
const int PWM_RESOLUTION = 8;
const int MOTOR_SPEED = 180;

void stopMotors() {
  ledcWrite(ENA, 0);
  ledcWrite(ENB, 0);

  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
}

void forward() {
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);

  ledcWrite(ENA, MOTOR_SPEED);
  ledcWrite(ENB, MOTOR_SPEED);
}

void backward() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);

  ledcWrite(ENA, MOTOR_SPEED);
  ledcWrite(ENB, MOTOR_SPEED);
}

void turnLeft() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);

  ledcWrite(ENA, MOTOR_SPEED);
  ledcWrite(ENB, MOTOR_SPEED);
}

void turnRight() {
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);

  ledcWrite(ENA, MOTOR_SPEED);
  ledcWrite(ENB, MOTOR_SPEED);
}

void setup() {
  Serial.begin(115200);

  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  ledcAttach(ENA, PWM_FREQ, PWM_RESOLUTION);
  ledcAttach(ENB, PWM_FREQ, PWM_RESOLUTION);

  stopMotors();

  Serial.println("GestureRobot Serial Controller Ready");
  Serial.println("F=Forward B=Backward L=Left R=Right S=Stop");
}

void loop() {
  if (Serial.available() > 0) {
    char command = toupper(Serial.read());

    switch (command) {
      case 'F':
        forward();
        Serial.println("Forward");
        break;

      case 'B':
        backward();
        Serial.println("Backward");
        break;

      case 'L':
        turnLeft();
        Serial.println("Left");
        break;

      case 'R':
        turnRight();
        Serial.println("Right");
        break;

      case 'S':
      default:
        stopMotors();
        Serial.println("Stop");
        break;
    }
  }
}
