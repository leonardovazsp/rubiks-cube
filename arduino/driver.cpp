const int xStep = 2;
const int yStep = 3;
const int zStep = 4;
const int xDir = 5;
const int yDir = 6;
const int zDir = 7;
const int enablePin = 12;
String inputString = ""; 
bool stringComplete = false;

void setup() {
  Serial.begin(9600);
  inputString.reserve(200);
  pinMode(xStep, OUTPUT);
  pinMode(yStep, OUTPUT);
  pinMode(zStep, OUTPUT);
  pinMode(xDir, OUTPUT);
  pinMode(yDir, OUTPUT);
  pinMode(zDir, OUTPUT);
  pinMode(enablePin, OUTPUT);
  digitalWrite(enablePin, LOW);

  digitalWrite(xStep, LOW);
  digitalWrite(yStep, LOW);
  digitalWrite(zStep, LOW);
  digitalWrite(xDir, LOW);
  digitalWrite(yDir, LOW);
  digitalWrite(zDir, LOW);
}

void loop() {
  if (stringComplete) {
    if (inputString == "get_device") {
      Serial.println("0");
      inputString = "";
      }
    else {
      executeCommand(inputString);
      Serial.println(inputString + ":ACK");
      inputString = "";
      stringComplete = false;
    }
  }
}

void serialEvent() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    if (inChar == '\n') {
      stringComplete = true;
    }
    else {inputString += inChar;}
  }
}
void executeCommand(String command) {
    if (command == "zcw") rotateMotor(zStep, zDir, true);
    else if (command == "zccw") rotateMotor(zStep, zDir, false);
    else if (command == "ycw") rotateMotor(yStep, yDir, true);
    else if (command == "yccw") rotateMotor(yStep, yDir, false);
    else if (command == "xcw") rotateMotor(xStep, xDir, true);
    else if (command == "xccw") rotateMotor(xStep, xDir, false);
}

void rotateMotor(int stepPin, int dirPin, bool clockwise) {
  if(clockwise){
    digitalWrite(dirPin, HIGH);
  }
  else {
    digitalWrite(dirPin, LOW);
  }
  for(int i = 0; i<50; i++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(1500);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(1500);
  }
}