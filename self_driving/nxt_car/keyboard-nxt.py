from nxt.bluesock import BlueSock
import nxt
import keyboard as key

ID = '00:16:53:17:EF:0A'  # MAC address NXT11

sock = BlueSock(ID)


def moveCar(bk):
    ''' remote control function to NXT robot with keyboard inputs.

    Args:
        bk (brick): brick object

    Returns:
        None - user stops exectution when pressing 'q'
    '''
    leftMotor = nxt.Motor(bk, nxt.PORT_B)
    rightMotor = nxt.Motor(bk, nxt.PORT_A)
    # both = nxt.SynchronizedMotors(leftMotor, rightMotor, 0)
    # rightboth = nxt.SynchronizedMotors(leftMotor, rightMotor, 100)
    # leftboth = nxt.SynchronizedMotors(rightMotor, leftMotor, 100)
    while True:

        try:
            if key.is_pressed('q'):
                print('Exiting...')
                break
            elif key.is_pressed('up'):
                rightMotor.weak_turn(20, 100)
                leftMotor.weak_turn(20, 100)
            elif key.is_pressed('down'):
                rightMotor.weak_turn(-20, 30)
                leftMotor.weak_turn(-20, 30)
            elif key.is_pressed('left'):
                rightMotor.weak_turn(20, 30)
                leftMotor.weak_turn(-20, 30)
            elif key.is_pressed('right'):
                rightMotor.weak_turn(-20, 30)
                leftMotor.weak_turn(20, 30)

            elif key.is_pressed('space'):
                leftMotor.idle()
                rightMotor.idle()
                leftMotor.brake()
                rightMotor.brake()
            else:
                pass
        except:
            break


if sock:
    # Connect to brick
    print('READY')
    brick = sock.connect()

    moveCar(brick)
    # Close socket
    sock.close()

# Failure
else:
    print 'No NXT bricks found'
