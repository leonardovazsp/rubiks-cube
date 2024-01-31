const int xStepPin = 2;
const int yStepPin = 3;
const int zStepPin = 4;
const int xDirPin = 5;
const int yDirPin = 6;
const int zDirPin = 7;
const int enablePin = 8;
const String device = "1";
String inputString = ""; 
bool stringComplete = false;
int globalDelay = 1000;

void setup() {
  Serial.begin(9600);
  inputString.reserve(200);
  pinMode(xStepPin, OUTPUT);
  pinMode(yStepPin, OUTPUT);
  pinMode(zStepPin, OUTPUT);
  pinMode(xDirPin, OUTPUT);
  pinMode(yDirPin, OUTPUT);
  pinMode(zDirPin, OUTPUT);
  pinMode(enablePin, OUTPUT);
  digitalWrite(enablePin, HIGH);
}

void loop() {
  if (stringComplete) {
    inputString.trim();
    if (inputString == "get_device") {
      Serial.println(device);
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

void rotateMotor(int stepPin, int dirPin, bool clockwise, int steps = 50) {
  digitalWrite(dirPin, clockwise ? HIGH : LOW);
  for(int i = 0; i < steps; i++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(globalDelay);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(globalDelay);
  }
}

void executeCommand(String command) {
    if (command.startsWith("rotate:")) {
        int colonIndex = command.indexOf(':');
        int commaIndex = command.indexOf(',');
        int secondCommaIndex = command.indexOf(',', commaIndex + 1);
    
        String axisString = command.substring(colonIndex + 1, commaIndex);
        String directionString = command.substring(commaIndex + 1, secondCommaIndex);
        String stepsString = command.substring(secondCommaIndex + 1);
    
        int stepsValue = stepsString.toInt();
    
        if (axisString == "x") rotateMotor(xStepPin, xDirPin, directionString == "cw", stepsValue);
        else if (axisString == "y") rotateMotor(yStepPin, yDirPin, directionString == "cw", stepsValue);
        else if (axisString == "z") rotateMotor(zStepPin, zDirPin, directionString == "cw", stepsValue);
    }
    else if (command == "activate") digitalWrite(enablePin, LOW);
    else if (command == "deactivate") digitalWrite(enablePin, HIGH);
    else if (command.startsWith("set_delay:")) {
      int colonIndex = command.indexOf(':');
      String delayString = command.substring(colonIndex + 1);

      int delayValue = delayString.toInt();

      setDelay(delayValue);
    }
}

void setDelay(int newDelay) {
  globalDelay = newDelay;
}