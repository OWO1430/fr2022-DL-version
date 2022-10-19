#include <LiquidCrystal_I2C.h>  //lcd
#include <LiquidCrystal.h>      //lcd
#include <Wire.h>               //lcd
#include <Stepper.h>
#include <Unistep2.h>
#include <Servo.h>   //servo
#include <SK6812.h>  //color botton
#include <Ultrasonic.h>
int rpwm, lpwm, sign, stepm, water, servom, color, perviousServom, runState, serOutput, stepperMove;
const int outputPinLoc[10] = { A0, 2, 3, 4, 5, 6, 7, 8, 9, A1 };      //A0 servo, 2~5 main motor, 6~9 step motor, 44 water relay
const int buttonPin[5] = { 34, 36, 38, 40, 42 };                      //34 state lock, 35~38 four states
const int motorPin[4] = { 2, 3, 4, 5 };                               //left forward, left backward, right forward, right backward
const int triangleTurnSpeed[6] = { 150, 150, 150, 150, 1700, 1700 };  //left forward, left backward, right forward, right backward, left turn time, right turn time
const int squareTurnSpeed[4] = { 150, 150, 2200, 2000 };              //left forward, right backward, turn time, forward time
const int circleTurnSpeed[3] = { 150, 150, 6000 };                    //left forward, right backward, turn time
const int clawAngle[2] = { 25, 100 };                                 //0 open, 1 close
const int grab_distance = 3;
String colorArray[5] = { "none", "red", "yellow", "blue", "black" };

SK6812 LED(3);
Stepper myStepper(2048, 6, 8, 7, 9);
// Unistep2 myStepper(6, 7, 8, 9, 4096, 900);
LiquidCrystal_I2C lcd(0x27, 16, 2);  //16,2為顯示器大小
Servo clawServo;
Ultrasonic ultrasonic(A3, A2);  //ultrasonic(trig, echo)
Ultrasonic bottomSonic(A5, A4); 

void readSerial();
void stopMotor();
void moveMotor(int left, int right);  //main moving code
void reverseMoveMotor(int left, int right);
void readRunState();
void readSign();               //0 none, 1 left triangle,2 right triangle, 3 square, 4 circle
void sign0();                  //move while checking claw code
void triangleTurn(bool type);  //triangle sign, true for left, false for right
void squareTurn(bool type);    //true = left; false = right
void circleTurn();             //180 degree turn
void readStateButton();        //manual change state
void printLcd(String str1, String str2);
void grabFruit();                    //0 off, 1 grab
void stepper();                      //0 stop, 1 forward, 2 backward
void watering();                     //0 off, 1 on
void colorDisplay();                 //0 off, 1 red, 2 yellow, 3 blue, 4 black
void buttonColorDisplay(int state);  //0 all white, 1~4 button color, 5 rainbow
void clearLED();


void setup() {
  for (int i = 0; i < 10; i++) pinMode(outputPinLoc[i], OUTPUT);
  digitalWrite(outputPinLoc[9], HIGH);
  for (int i = 0; i < 5; i++) pinMode(buttonPin[i], INPUT_PULLUP);  //for (int i = 0; i < 5; i++) pinMode(buttonPin[i], INPUT);
  Serial.begin(9600);
  lcd.init();
  lcd.backlight();
  LED.set_output(44);
  clearLED();
  clawServo.attach(A0);
  clawServo.write(clawAngle[0]);
  // myStepper.run();
  myStepper.setSpeed(15);

}

void loop() {
  readStateButton();
  readSerial();
  readRunState();
}

void readStateButton() {
  int stateInput = 0;
  if (digitalRead(buttonPin[0]) == LOW) {
    stopMotor();
    printLcd("STATE UNLOCK", "PLZ INPUT STATE");
    buttonColorDisplay(0);
    while (digitalRead(buttonPin[0]) == LOW) {  //while unlock
      for (int i = 1; i < 5; i++) {
        if (digitalRead(buttonPin[i]) == LOW) {
          printLcd("INPUT STATE " + (String)i, "COMFIRM?");
          buttonColorDisplay(i);
          stateInput = i;
        }
      }
    }
    printLcd("    STATE " + (String)stateInput, "===ACTIVATED===");
    buttonColorDisplay(stateInput + 4);
    lcd.clear();
    Serial.println(stateInput);
  } else {
    Serial.println(serOutput);
  }
}

void readSerial() {
  while (true) {
    if (Serial.available()) {
      char buffer[6];
      String str = Serial.readStringUntil('\n');
      str.toCharArray(buffer, 7);
      if (buffer[0] == '9') {
        printLcd("task", str);
        runState = 1;
        sign = buffer[1] - '0';
        stepm = buffer[2] - '0';
        water = buffer[3] - '0';
        servom = buffer[4] - '0';
        color = buffer[5] - '0';
        serOutput = 999;
        // printLcd("state 1 /s" + String(sign), "st" + String(stepm) + "/w" + String(water) + "/sv" + String(servom) + "/c" + String(color));
      } else {
        if (buffer[0] == '8') {
          runState = 3;
          lpwm = 30;
          rpwm = 30;
        } else {
          runState = 2;
          lpwm = (buffer[0] - '0') * 100 + (buffer[1] - '0') * 10 + (buffer[2] - '0');
          rpwm = (buffer[3] - '0') * 100 + (buffer[4] - '0') * 10 + (buffer[5] - '0');
        }
        lcd.clear();
        lcd.setCursor(0, 0);
        lcd.print(lpwm);
        lcd.setCursor(0, 1);
        lcd.print(rpwm);
        serOutput = lpwm;
      }
      break;
    }
  }
}

void readRunState() {
  if (runState == 1) readSign();
  else if (runState == 2) moveMotor(lpwm, rpwm);
  else if (runState == 3) reverseMoveMotor(lpwm, rpwm);
}

void readSign() {
  switch (sign) {  //0 none, 1 left triangle,2 right triangle, 3 square, 4 circle
    case 0:
      sign0();
      break;
    case 1:
      triangleTurn(true);
      break;
    case 2:
      triangleTurn(false);
      break;
    case 3:
      squareTurn(false);
      break;
    default:
      circleTurn();
      break;
  }
}

void sign0() {
  stopMotor();
  grabFruit();
  stepper();
  watering();
  colorDisplay();
}

void stopMotor() {
  for (int i = 0; i < 4; i++) analogWrite(motorPin[i], 0);
}

void moveMotor(int left, int right) {  //main moving code
  analogWrite(motorPin[0], left);
  analogWrite(motorPin[2], right);
  analogWrite(motorPin[1], 0);
  analogWrite(motorPin[3], 0);
}

void reverseMoveMotor(int left, int right) {  //main moving code
  analogWrite(motorPin[0], 0);
  analogWrite(motorPin[2], 0);
  analogWrite(motorPin[1], left);
  analogWrite(motorPin[3], right);
}

void triangleTurn(bool type) {  //triangle sign, true for left, false for right
  if (type) {                   //turn left
    squareTurn(true);
    moveMotor(128, 128);
    delay(6500);
    stopMotor();
    analogWrite(motorPin[1], triangleTurnSpeed[1]);
    analogWrite(motorPin[2], triangleTurnSpeed[2]);
    delay(triangleTurnSpeed[4]);
  } else {  //turn right
    squareTurn(false);
    moveMotor(128, 128);
    delay(6500);
    stopMotor();
    analogWrite(motorPin[0], triangleTurnSpeed[0]);
    analogWrite(motorPin[3], triangleTurnSpeed[3]);
    delay(triangleTurnSpeed[5]);
  }
  stopMotor();
}

void squareTurn(bool type) {  //true = left; false = right
  stopMotor();
  if (type) {
    analogWrite(motorPin[0], squareTurnSpeed[0]);
    analogWrite(motorPin[3], squareTurnSpeed[1]);
    delay(squareTurnSpeed[2]);
  } else {
    analogWrite(motorPin[1], squareTurnSpeed[0]);
    analogWrite(motorPin[2], squareTurnSpeed[1]);
    delay(squareTurnSpeed[2]);
  }
  stopMotor();
}

void circleTurn() {  //180 degree turn
  stopMotor();
  analogWrite(motorPin[1], circleTurnSpeed[0]);
  analogWrite(motorPin[2], circleTurnSpeed[1]);
  delay(circleTurnSpeed[2]);
  stopMotor();
}

void printLcd(String str1, String str2) {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print(str1);
  lcd.setCursor(0, 1);
  lcd.print(str2);
}

void stepper() {
  if (stepm == 1) {
    myStepper.step(100);
    stepperMove += 1;
  } else if (stepm == 2) {
    myStepper.step(-100);
    stepperMove -= 1;
  }
}

void grabFruit() {
  if (servom) {
    while (ultrasonic.read() > grab_distance) {
      myStepper.step(100);
      stepperMove ++;
    }
    printLcd("task", "grab finish");
    clawServo.write(clawAngle[1]);
    myStepper.step(-100 * stepperMove);
    moveMotor(50, 50);
    delay(3000);
    stopMotor();
    myStepper.step(100 * stepperMove);
    clawServo.write(clawAngle[0]);
    myStepper.step(-100 * stepperMove);
  }
}

void watering() {  //0 off, 1 on
  if (water == 0) digitalWrite(outputPinLoc[9], HIGH);
  else {
    digitalWrite(outputPinLoc[9], LOW); 
    delay(50000);
    digitalWrite(outputPinLoc[9], HIGH);
    myStepper.step(-100*stepperMove);
    stepperMove = 0;
    delay(1000);
    circleTurn();
  }
}

void colorDisplay() {  //0 off, 1 red, 2 yellow, 3 blue, 4 black
  if (color != 0) {
    if (color < 4){ //fruit
      printLcd("The color is:", colorArray[color]);
      delay(5000);
      lcd.clear();
      moveMotor(128, 128);
      delay(1500);
        //tba
      while(bottomSonic.read()>20){
        myStepper.step(-100);
        stepperMove--;
      } 
    }else{  //water
      printLcd("The color is:", colorArray[color-4]);
      delay(5000);
      lcd.clear();
      moveMotor(100, 100);
      delay(3000);  // tba
      myStepper.step(10000);  //tba
    }
    
  }
}

void clearLED() {
  for (int i = 0; i < 3; i++) LED.set_rgbw(i, { 0, 0, 0, 0 });
  LED.sync();
  delay(50);
}

void buttonColorDisplay(int state) {
  switch (state) {
    default:
      LED.set_rgbw(0, { 255, 255, 255, 255 });
      LED.set_rgbw(1, { 255, 255, 255, 255 });
      LED.set_rgbw(2, { 255, 255, 255, 255 });
      LED.sync();
      break;
    case 1:
      LED.set_rgbw(0, { 51, 102, 204, 255 });
      LED.set_rgbw(1, { 255, 255, 255, 255 });
      LED.set_rgbw(2, { 255, 255, 255, 255 });
      LED.sync();
      delay(100);
      LED.set_rgbw(0, { 255, 255, 255, 51 });
      LED.set_rgbw(1, { 102, 204, 255, 255 });
      LED.set_rgbw(2, { 255, 255, 255, 255 });
      LED.sync();
      delay(100);
      LED.set_rgbw(0, { 255, 255, 255, 255 });
      LED.set_rgbw(1, { 255, 255, 51, 102 });
      LED.set_rgbw(2, { 204, 255, 255, 255 });
      LED.sync();
      delay(100);
      LED.set_rgbw(0, { 255, 255, 255, 255 });
      LED.set_rgbw(1, { 255, 255, 255, 255 });
      LED.set_rgbw(2, { 255, 51, 102, 204 });
      LED.sync();
      delay(100);
      LED.set_rgbw(0, { 255, 255, 255, 255 });
      LED.set_rgbw(1, { 255, 255, 51, 102 });
      LED.set_rgbw(2, { 204, 255, 255, 255 });
      LED.sync();
      delay(100);
      LED.set_rgbw(0, { 255, 255, 255, 51 });
      LED.set_rgbw(1, { 102, 204, 255, 255 });
      LED.set_rgbw(2, { 255, 255, 255, 255 });
      LED.sync();
      delay(100);
      LED.set_rgbw(0, { 51, 102, 204, 255 });
      LED.set_rgbw(1, { 255, 255, 255, 255 });
      LED.set_rgbw(2, { 255, 255, 255, 255 });
      LED.sync();
      break;
    case 2:
      LED.set_rgbw(0, { 255, 255, 255, 230 });
      LED.set_rgbw(1, { 51, 204, 255, 255 });
      LED.set_rgbw(2, { 255, 255, 255, 255 });
      LED.sync();
      delay(100);
      LED.set_rgbw(0, { 255, 255, 255, 255 });
      LED.set_rgbw(1, { 255, 255, 230, 102 });
      LED.set_rgbw(2, { 204, 255, 255, 255 });
      LED.sync();
      delay(100);
      LED.set_rgbw(0, { 255, 255, 255, 255 });
      LED.set_rgbw(1, { 255, 255, 255, 255 });
      LED.set_rgbw(2, { 255, 230, 102, 204 });
      LED.sync();
      delay(100);
      LED.set_rgbw(0, { 255, 255, 255, 255 });
      LED.set_rgbw(1, { 255, 255, 230, 102 });
      LED.set_rgbw(2, { 204, 255, 255, 255 });
      LED.sync();
      delay(100);
      LED.set_rgbw(0, { 255, 255, 255, 230 });
      LED.set_rgbw(1, { 51, 204, 255, 255 });
      LED.set_rgbw(2, { 255, 255, 255, 255 });
      LED.sync();
      delay(100);
      LED.set_rgbw(0, { 230, 102, 204, 255 });
      LED.set_rgbw(1, { 255, 255, 255, 255 });
      LED.set_rgbw(2, { 255, 255, 255, 255 });
      LED.sync();
      delay(100);
      LED.set_rgbw(0, { 255, 255, 255, 230 });
      LED.set_rgbw(1, { 51, 204, 255, 255 });
      LED.set_rgbw(2, { 255, 255, 255, 255 });
      LED.sync();
      break;
    case 3:
      LED.set_rgbw(0, { 255, 255, 255, 255 });
      LED.set_rgbw(1, { 255, 255, 34, 178 });
      LED.set_rgbw(2, { 34, 255, 255, 255 });
      LED.sync();
      delay(100);
      LED.set_rgbw(0, { 255, 255, 255, 255 });
      LED.set_rgbw(1, { 255, 255, 255, 255 });
      LED.set_rgbw(2, { 255, 34, 178, 34 });
      LED.sync();
      delay(100);
      LED.set_rgbw(0, { 255, 255, 255, 255 });
      LED.set_rgbw(1, { 255, 255, 34, 178 });
      LED.set_rgbw(2, { 34, 255, 255, 255 });
      LED.sync();
      delay(100);
      LED.set_rgbw(0, { 255, 255, 255, 34 });
      LED.set_rgbw(1, { 178, 34, 255, 255 });
      LED.set_rgbw(2, { 255, 255, 255, 255 });
      LED.sync();
      delay(100);
      LED.set_rgbw(0, { 34, 178, 34, 255 });
      LED.set_rgbw(1, { 255, 255, 255, 255 });
      LED.set_rgbw(2, { 255, 255, 255, 255 });
      LED.sync();
      delay(100);
      LED.set_rgbw(0, { 255, 255, 255, 34 });
      LED.set_rgbw(1, { 178, 34, 255, 255 });
      LED.set_rgbw(2, { 255, 255, 255, 255 });
      LED.sync();
      delay(100);
      LED.set_rgbw(0, { 255, 255, 255, 255 });
      LED.set_rgbw(1, { 255, 255, 34, 178 });
      LED.set_rgbw(2, { 34, 255, 255, 255 });
      LED.sync();
      break;
    case 4:
      LED.set_rgbw(0, { 255, 255, 255, 255 });
      LED.set_rgbw(1, { 255, 255, 255, 255 });
      LED.set_rgbw(2, { 255, 134, 184, 11 });
      LED.sync();
      delay(100);
      LED.set_rgbw(0, { 255, 255, 255, 255 });
      LED.set_rgbw(1, { 255, 255, 134, 184 });
      LED.set_rgbw(2, { 11, 255, 255, 255 });
      LED.sync();
      delay(100);
      LED.set_rgbw(0, { 255, 255, 255, 134 });
      LED.set_rgbw(1, { 184, 11, 255, 255 });
      LED.set_rgbw(2, { 255, 255, 255, 255 });
      LED.sync();
      delay(100);
      LED.set_rgbw(0, { 134, 184, 11, 255 });
      LED.set_rgbw(1, { 255, 255, 255, 255 });
      LED.set_rgbw(2, { 255, 255, 255, 255 });
      LED.sync();
      delay(100);
      LED.set_rgbw(0, { 255, 255, 255, 134 });
      LED.set_rgbw(1, { 184, 11, 255, 255 });
      LED.set_rgbw(2, { 255, 255, 255, 255 });
      LED.sync();
      delay(100);
      LED.set_rgbw(0, { 255, 255, 255, 255 });
      LED.set_rgbw(1, { 255, 255, 134, 184 });
      LED.set_rgbw(2, { 11, 255, 255, 255 });
      LED.sync();
      delay(100);
      LED.set_rgbw(0, { 255, 255, 255, 255 });
      LED.set_rgbw(1, { 255, 255, 255, 255 });
      LED.set_rgbw(2, { 255, 134, 184, 11 });
      LED.sync();
      break;
    case 5:
      LED.set_rgbw(0, { 51, 102, 204, 255 });
      LED.set_rgbw(1, { 255, 255, 255, 255 });
      LED.set_rgbw(2, { 255, 255, 255, 255 });
      LED.sync();
      delay(100);
      LED.set_rgbw(0, { 51, 102, 204, 51 });
      LED.set_rgbw(1, { 102, 204, 255, 255 });
      LED.set_rgbw(2, { 255, 255, 255, 255 });
      LED.sync();
      delay(100);
      LED.set_rgbw(0, { 51, 102, 204, 51 });
      LED.set_rgbw(1, { 102, 204, 51, 102 });
      LED.set_rgbw(2, { 204, 255, 255, 255 });
      LED.sync();
      delay(100);
      LED.set_rgbw(0, { 51, 102, 204, 51 });
      LED.set_rgbw(1, { 102, 204, 51, 102 });
      LED.set_rgbw(2, { 204, 51, 102, 204 });
      LED.sync();
      delay(1700);
      clearLED();
      break;
    case 6:
      LED.set_rgbw(0, { 255, 255, 255, 230 });
      LED.set_rgbw(1, { 51, 204, 255, 255 });
      LED.set_rgbw(2, { 255, 255, 255, 255 });
      LED.sync();
      delay(150);
      LED.set_rgbw(0, { 230, 102, 204, 230 });
      LED.set_rgbw(1, { 51, 204, 230, 102 });
      LED.set_rgbw(2, { 204, 255, 255, 255 });
      LED.sync();
      delay(150);
      LED.set_rgbw(0, { 230, 102, 204, 230 });
      LED.set_rgbw(1, { 51, 204, 230, 102 });
      LED.set_rgbw(2, { 204, 230, 102, 204 });
      LED.sync();
      delay(1700);
      clearLED();
      break;
    case 7:
      LED.set_rgbw(0, { 255, 255, 255, 255 });
      LED.set_rgbw(1, { 255, 255, 34, 178 });
      LED.set_rgbw(2, { 34, 255, 255, 255 });
      LED.sync();
      delay(150);
      LED.set_rgbw(0, { 255, 255, 255, 34 });
      LED.set_rgbw(1, { 178, 34, 34, 178 });
      LED.set_rgbw(2, { 34, 34, 178, 34 });
      LED.sync();
      delay(150);
      LED.set_rgbw(0, { 34, 178, 34, 34 });
      LED.set_rgbw(1, { 178, 34, 34, 178 });
      LED.set_rgbw(2, { 34, 34, 178, 34 });
      LED.sync();
      delay(1700);
      clearLED();
      break;
    case 8:
      LED.set_rgbw(0, { 255, 255, 255, 255 });
      LED.set_rgbw(1, { 255, 255, 255, 255 });
      LED.set_rgbw(2, { 255, 134, 184, 11 });
      LED.sync();
      delay(100);
      LED.set_rgbw(0, { 255, 255, 255, 255 });
      LED.set_rgbw(1, { 255, 255, 134, 184 });
      LED.set_rgbw(2, { 11, 134, 184, 11 });
      LED.sync();
      delay(100);
      LED.set_rgbw(0, { 255, 255, 255, 134 });
      LED.set_rgbw(1, { 184, 11, 134, 184 });
      LED.set_rgbw(2, { 11, 134, 184, 11 });
      LED.sync();
      delay(100);
      LED.set_rgbw(0, { 134, 184, 11, 134 });
      LED.set_rgbw(1, { 184, 11, 134, 184 });
      LED.set_rgbw(2, { 11, 134, 184, 11 });
      LED.sync();
      delay(1700);
      clearLED();
      break;
  }
}
