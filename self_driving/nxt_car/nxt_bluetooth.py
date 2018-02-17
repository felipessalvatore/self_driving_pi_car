from nxt import bluesock

NXT_ID = '00:16:53:17:EF:0A'  # MAC address NXT11


def connectCar(blue_id=NXT_ID):
    '''
    Connect to nxt brick using the id 'blue_id'. The default
    id is the global variable NXT_ID. It returs one
    object for bluethoot connection 'sock' and one
    object for NXT control 'brick'.

    :param blue_id: Bluethoot MAC address
    :type blue_id: str
    :rtype: (nxt.bluesock.BlueSock, nxt.brick)
    '''
    try:
        sock = bluesock.BlueSock(blue_id)
        brick = sock.connect()
        return sock, brick
    except:
        print("NO connection with {}".format(NXT_ID))


def disconnectCar(open_sock):
    '''
    Disconnect from NXT-car

    :param open_sock: opened Bluetooth socket communication
    :type open_sock: nxt.bluesock.BlueSock
    '''
    open_sock.close()
