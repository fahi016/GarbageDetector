/*
 * esp32_arm_controller.ino
 *
 * Drives a 6-DOF robotic arm (6 hobby servos) from angle commands streamed
 * over Serial by the MuJoCo simulation. The ESP32 does NOT re-plan or smooth
 * the trajectory -- it applies every received frame to the servos the
 * instant it is parsed, with no delay() anywhere in the receive path. All
 * smoothing/timing comes from how often and how finely the sender emits
 * frames; the firmware's only job is: parse fast, clamp to safe limits,
 * write immediately.
 *
 * Library required (Arduino IDE -> Tools -> Manage Libraries):
 *   "ESP32Servo" by Kevin Harrington / John K. Bennett
 *
 * ---------------------------------------------------------------------
 * Serial protocol
 * ---------------------------------------------------------------------
 * One frame = one line:
 *
 *     <base,shoulder,elbow,wrist,wristRot,gripper>
 *
 *   - 6 comma-separated angles in DEGREES, integer or decimal, in this
 *     fixed order (matches the joint table in the project spec).
 *   - Framed with '<' ... '>' so a partial/garbled line (e.g. the sender
 *     starting mid-line, or noise) is dropped instead of misread.
 *   - A trailing '\n' is optional -- it is simply ignored while the parser
 *     waits for the next '<'.
 *
 * Example frames:
 *     <90,90,90,90,90,90>          // home / neutral pose
 *     <45,120,60,90,90,180>        // gripper closed
 *     <0,0,160,20,190,50>          // deliberately out-of-range test:
 *                                  // clamped on-board to <2,0,160,45,180,90>
 *
 * Baud rate: 115200 (set the same rate on the sending side).
 *
 * ---------------------------------------------------------------------
 * Servo limits (enforced on-board -- values are ALWAYS clamped before
 * being written to a servo, regardless of what the sender sends)
 * ---------------------------------------------------------------------
 *   Servo 1  Base            2   - 179
 *   Servo 2  Shoulder        0   - 180
 *   Servo 3  Elbow           0   - 180
 *   Servo 4  Wrist           45  - 145
 *   Servo 5  Wrist Rotation  0   - 180
 *   Servo 6  Gripper         90  - 180   (90 = open, 180 = closed)
 */

#include <ESP32Servo.h>

// ---------------------------------------------------------------------
// Pin assignment -- change these to match your wiring. Pick PWM-capable
// GPIOs and avoid strapping pins (0, 2, 15) and input-only pins (34-39).
// ---------------------------------------------------------------------
const uint8_t SERVO_PIN[6] = {13, 12, 14, 27, 26, 25};

// ---------------------------------------------------------------------
// Per-servo allowed angle range, in the fixed joint order:
// 0 base, 1 shoulder, 2 elbow, 3 wrist, 4 wristRot, 5 gripper
// ---------------------------------------------------------------------
const int SERVO_MIN[6] = {2, 0, 0, 45, 0, 90};
const int SERVO_MAX[6] = {179, 180, 180, 145, 180, 180};

// Safe boot pose: mid-range on every joint, gripper open. All values are
// inside every joint's own limits above.
const int HOME_POSE[6] = {90, 90, 90, 90, 90, 90};

Servo servos[6];

// ---------------------------------------------------------------------
// Non-blocking frame reader: accumulates bytes between '<' and '>' with
// no delay()/blocking read, so it never stalls the loop waiting on Serial.
// ---------------------------------------------------------------------
const size_t FRAME_BUF_SIZE = 64;
char frameBuf[FRAME_BUF_SIZE];
size_t frameLen = 0;
bool receiving = false;

int clampToServoRange(uint8_t idx, float angle) {
  int a = (int)lroundf(angle);
  if (a < SERVO_MIN[idx]) return SERVO_MIN[idx];
  if (a > SERVO_MAX[idx]) return SERVO_MAX[idx];
  return a;
}

// Parses "a,b,c,d,e,f" (already stripped of '<' '>') into 6 floats.
// Returns true only if all 6 fields were present and numeric.
bool parseFrame(char *buf, float out[6]) {
  uint8_t field = 0;
  char *tok = strtok(buf, ",");
  while (tok != NULL && field < 6) {
    char *endPtr = NULL;
    float v = strtof(tok, &endPtr);
    if (endPtr == tok) {
      return false;  // token wasn't a number
    }
    out[field++] = v;
    tok = strtok(NULL, ",");
  }
  return field == 6 && tok == NULL;
}

void applyFrame(const float angles[6]) {
  for (uint8_t i = 0; i < 6; i++) {
    int clamped = clampToServoRange(i, angles[i]);
    servos[i].write(clamped);
  }
}

void handleByte(char c) {
  if (c == '<') {
    // Start (or restart) a frame -- always resync on '<', which also
    // recovers cleanly from a previously truncated/garbled line.
    receiving = true;
    frameLen = 0;
    return;
  }
  if (!receiving) {
    return;  // ignore anything outside a frame (e.g. '\n', '\r', noise)
  }
  if (c == '>') {
    frameBuf[frameLen] = '\0';
    float angles[6];
    if (parseFrame(frameBuf, angles)) {
      applyFrame(angles);
    }
    receiving = false;
    frameLen = 0;
    return;
  }
  if (frameLen < FRAME_BUF_SIZE - 1) {
    frameBuf[frameLen++] = c;
  } else {
    // Frame too long / malformed -- drop it and wait for the next '<'.
    receiving = false;
    frameLen = 0;
  }
}

void setup() {
  Serial.begin(115200);

  // ESP32Servo needs the LEDC PWM timers allocated before attach().
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);

  for (uint8_t i = 0; i < 6; i++) {
    servos[i].setPeriodHertz(50);            // standard analog servo refresh
    servos[i].attach(SERVO_PIN[i], 500, 2400); // full 0-180 pulse-width range
    servos[i].write(HOME_POSE[i]);
  }
}

void loop() {
  // Drain everything currently buffered by the UART, byte by byte,
  // with no delay() call anywhere in this path.
  while (Serial.available() > 0) {
    handleByte((char)Serial.read());
  }
}
