from learning.basics.yoloImage import yoloImage
from learning.basics.yoloCamera import yoloCamera

def callOtherFunction():
    # yoloImage('Images/1.png', 'YoloWeight/yolov8l.pt')
    yoloCamera('YoloWeight/yolov8l.pt')
    


# Main function
if __name__ == '__main__':
    callOtherFunction()