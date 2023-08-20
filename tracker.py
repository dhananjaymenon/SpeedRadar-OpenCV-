import cv2
import math
import time
import numpy as np
import os

"""
The below values are specific to the video used
These values will have to be changed for a different video
"""
limit = 80  # km/hr
start_line = 410  # pixel value to start the timer
end_line = 235  # pixel value to stop the timer
buffer = 20  # pixels (recommended 15 - 25 pixels for proper detection)


traffic_record_folder_name = "TrafficRecord"

if not os.path.exists(traffic_record_folder_name):
    os.makedirs(traffic_record_folder_name)
    os.makedirs(traffic_record_folder_name+"//exceeded")


speed_record_file_location = traffic_record_folder_name + "//SpeedRecord.txt"
file = open(speed_record_file_location, "w")
file.write("ID \t SPEED\n------\t-------\n")
file.close()


class EuclideanDistTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}

        self.id_count = 0
        # self.start = 0
        # self.stop = 0
        self.et = 0
        self.s1 = np.zeros((1, 1000))
        self.s2 = np.zeros((1, 1000))
        self.s = np.zeros((1, 1000))
        self.f = np.zeros(1000)
        self.capf = np.zeros(1000)
        self.count = 0
        self.exceeded = 0

    def update(self, objects_rect):
        """
        Updates an already existing objects coordinates and
        assigns a new ID to a new detected object
        """
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # CHECK IF OBJECT IS DETECTED ALREADY
            same_object_detected = False

            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 70:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True

                    # START TIMER
                    if start_line <= y <= start_line + buffer:
                        self.s1[0, id] = time.time()

                    # STOP TIMER and FIND DIFFERENCE
                    if end_line <= y <= end_line + buffer:
                        self.s2[0, id] = time.time()
                        self.s[0, id] = self.s2[0, id] - self.s1[0, id]

                    # CAPTURE FLAG
                    if y < 235:
                        self.f[id] = 1

            # NEW OBJECT DETECTION
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1
                self.s[0, self.id_count] = 0
                self.s1[0, self.id_count] = 0
                self.s2[0, self.id_count] = 0

        # ASSIGN NEW ID to OBJECT
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        self.center_points = new_center_points.copy()
        return objects_bbs_ids

    # SPEEED FUNCTION
    def getsp(self, id: int):
        """
        Speed is calculated based on the time taken for vehicle to cross
        a segment of road.
        the constant 214.15 depends on the video
        """
        if self.s[0, id] != 0:
            s = 214.15 / self.s[0, id]
        else:
            s = 0

        return int(s)

    # SAVE VEHICLE DATA
    def capture(self, img, x, y, h, w, sp, id):
        """
        Image of a vehicle with its recorded speed is stored in a file
        """
        if self.capf[id] == 0:
            self.capf[id] = 1
            self.f[id] = 0
            crop_img = img[y - 5:y + h + 5, x - 5:x + w + 5]
            n = str(id) + "_speed_" + str(sp)
            file = traffic_record_folder_name + '//' + n + '.jpg'
            cv2.imwrite(file, crop_img)
            self.count += 1
            filet = open(speed_record_file_location, "a")
            if sp > limit:
                file2 = traffic_record_folder_name + '//exceeded//' + n + '.jpg'
                cv2.imwrite(file2, crop_img)
                filet.write(str(id) + " \t " + str(sp) + "<---exceeded\n")
                self.exceeded += 1
            else:
                filet.write(str(id) + " \t " + str(sp) + "\n")
            filet.close()

    # SPEED_LIMIT
    def limit(self):
        return limit

    # TEXT FILE SUMMARY
    def end(self):
        """
        At the end of the video, a summary of vehicles and their speeds are displayed
        in a text file
        :return:
        """
        file = open(speed_record_file_location, "a")
        file.write("\n-------------\n")
        file.write("-------------\n")
        file.write("SUMMARY\n")
        file.write("-------------\n")
        file.write("Total Vehicles :\t" + str(self.count) + "\n")
        file.write("Exceeded speed limit :\t" + str(self.exceeded))
        file.close()
