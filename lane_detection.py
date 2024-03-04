import cv2
from laneprocessing.LaneProcessing import ClusterLane

if __name__ == "__main__":
    import time
    # read data from camera
    # video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # read data from video file
    video = cv2.VideoCapture("./data/video_3_3_2024_full_map.mp4")

    # Init Lane Process Class
    lane_processor = ClusterLane(
                                one_lane_bias = 120, 
                                center_image = 320, 
                                num_points_to_center = 11, 
                                y_middle_point = 10,
                                top_crop = 280,
                                bot_crop = 480,
                                draw_line = True
                                )

    # Init Video Record
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('video_3_3_2024_dust_lane.avi', fourcc, 30.0, (640,480))

    while True:
        t1 = time.time()
        ret, frame = video.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640,480))
        
        # Write frame to video
        # out.write(frame)

        # Process Lane
        lane_processor.cluster_lane(frame)

        # Get data processed
        image_out = lane_processor.img
        middle_point = lane_processor.middle_point

        print("FPS: ", 1/(time.time() - t1))
        # Check Intersection
        print("/////////")
        print("Intersection", lane_processor.get_intersection())

        # Draw middle point
        cv2.circle(image_out,(middle_point[0], middle_point[1] ), 5, (188, 144, 255), thickness=7, lineType=cv2.LINE_AA)

        # Show output image
        cv2.imshow("image", frame)
        cv2.imshow("image out", image_out)
        cv2.waitKey(0)