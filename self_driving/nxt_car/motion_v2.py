'''
Uses USB connection (requires pyusb and udev rules for lego USB)
Tries to increase velocity using synchronized motors for forward and backward motion.
Is possible to increase power up to 60 in weak_turn() calls, but it best behaves with 40
'''
import nxt
import time
import keyboard as key
brick = nxt.locator.find_one_brick()
brick.play_tone_and_wait(440, 1000) # brick is alive and connected to USB

# initialize and reset position
leftMotor = nxt.Motor(brick, nxt.PORT_B)
rightMotor = nxt.Motor(brick, nxt.PORT_A)
leftMotor.reset_position(False)
rightMotor.reset_position(False)
both = nxt.SynchronizedMotors(leftMotor, rightMotor, 0)
while True:
        try:
            if key.is_pressed('q'):
                print('Exiting...')
                break
            elif key.is_pressed('up'):
                #rightMotor.weak_turn(20, 100)
                #leftMotor.weak_turn(20, 100)
                 both.run(40)
		         time.sleep(0.05)
            elif key.is_pressed('down'):
                #rightMotor.weak_turn(-20,30)
                #leftMotor.weak_turn(-20,30)
                 both.run(-40)
		        time.sleep(0.05)
            elif key.is_pressed('left'):
                rightMotor.weak_turn(60,30)
                leftMotor.weak_turn(-60,30)
            elif key.is_pressed('right'):
                rightMotor.weak_turn(-60,30)
                leftMotor.weak_turn(60,30)
                # leftMotor.run(-40)
                # rightMotor.run(40)

            elif key.is_pressed('space'):
                leftMotor.idle()
                rightMotor.idle()
                leftMotor.brake()
                rightMotor.brake()
                # both.idle()
                # both.brake()
                print('stopping...')
            else:
		leftMotor.idle()
		rightMotor.idle()
                pass
        except:
		break



#rightMotor.weak_turn(-20,turnDegrees)
#leftMotor.weak_turn(20,turnDegrees)
#time.sleep(3) # wait for the motors to stop

#print abs(rightMotor.get_tacho().tacho_count)
#print abs(leftMotor.get_tacho().tacho_count)


