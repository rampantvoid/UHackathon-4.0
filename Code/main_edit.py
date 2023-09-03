import yolov5
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
import time
import random
# pass these when calling the function
areaArr = [[(430,67),(111,488),(962,478),(610,61)],[(626,246),(163,490),(731,487),(718,254)],[(362,145),(242,485),(927,485),(618,135)],[(608,209),(365,490),(775,495),(715,189)]]
videofiles = ['intersection1.mp4','intersection2.mp4','intersection3.mp4','intersection4empty.mp4']


class TrafficSignalController:
    def __init__(self):
        self.signal_timings = {}  # Store signal timings for each lane/intersection
        self.signal_duration = 60  # Signal duration in seconds
        self.green_light_duration = 30  # Green light duration in seconds
        self.yellow_light_duration = 5  # Yellow light duration in seconds

        # ACO parameters
        self.num_ants = 10
        self.pheromone_matrix = None
        self.alpha = 1.0  # Pheromone importance
        self.beta = 2.0   # Heuristic information importance
        self.evaporation_rate = 0.1
        self.q0 = 0.9      # Exploration factor

    def initialize_pheromone_matrix(self):
        # Initialize the pheromone matrix with some initial values
        num_lanes = len(self.signal_timings["Intersection1"])-1
        self.pheromone_matrix = np.ones((num_lanes, num_lanes))

    def aco_optimize(self):
        self.initialize_pheromone_matrix()

        for _ in range(self.signal_duration):
            # Generate solutions for each ant
            solutions = []
            for _ in range(self.num_ants):
                ant_solution = self.generate_ant_solution()
                solutions.append(ant_solution)

            # Update pheromone matrix based on ant solutions
            self.update_pheromone_matrix(solutions)

            # Update green light durations based on pheromone matrix
            self.update_green_light_durations()
    
    def generate_ant_solution(self):
        num_lanes = len(self.signal_timings["Intersection1"])
        ant_solution = []
        
        for _ in range(num_lanes):
            # Implement ant's solution construction logic here
            # For simplicity, we'll choose a random duration between 10 and 30
            green_duration = random.randint(10, 30)
            ant_solution.append(green_duration)
        
        return ant_solution

    def update_pheromone_matrix(self, solutions):
        for i in range(len(solutions)):
            for j in range(len(solutions[i])):
                # Update pheromone based on ant's solution quality
                pheromone_delta = 1 / (1 + solutions[i][j])
                self.pheromone_matrix[i][j] = (1 - self.evaporation_rate) * self.pheromone_matrix[i][j] + pheromone_delta
    
    def update_green_light_durations(self):
        num_lanes = len(self.signal_timings["Intersection1"])
        
        # Calculate probabilities for each lane based on pheromone levels
        total_pheromone = np.sum(self.pheromone_matrix)
        lane_probabilities = [self.pheromone_matrix[i][j] / total_pheromone for i in range(num_lanes) for j in range(num_lanes)]
        
        # Choose a lane with a probability proportional to pheromone levels
        selected_lane = np.random.choice(num_lanes, p=lane_probabilities)
        
        # Update green light durations based on the selected lane
        for i, lane in enumerate(self.signal_timings["Intersection1"]):
            if i == selected_lane:
                self.signal_timings["Intersection1"][lane]["green_duration"] += 1
            else:
                self.signal_timings["Intersection1"][lane]["green_duration"] -= 1
    

    def add_intersection(self, intersection_name, lanes, initial_timings):
        # Initialize signal timings for each lane at the intersection
        intersection = {}
        for lane, timing in zip(lanes, initial_timings):
            intersection[lane] = {
                "green_duration": timing,
                "yellow_duration": self.yellow_light_duration,
                "current_lane": None,
            }
        self.signal_timings[intersection_name] = intersection

    def update_signal(self, intersection_name, lane, new_green_duration, new_yellow_duration, current_lane):
        self.signal_timings[intersection_name][lane]["green_duration"] = new_green_duration
        self.signal_timings[intersection_name][lane]["yellow_duration"] = new_yellow_duration
        self.signal_timings[intersection_name][lane]["current_lane"] = current_lane

    def adjust_signals(self, intersection_name, yolo_output):
        intersection = self.signal_timings[intersection_name]
        lanes = list(intersection.keys())
        for i, lane in enumerate(lanes):
            current_green_duration = intersection[lane]["green_duration"]
            current_yellow_duration = intersection[lane]["yellow_duration"]
            
          
            vehicles_detected = yolo_output[i]
            
           
            traffic_category = self.categorize_traffic(vehicles_detected)
            
            
            new_green_duration, new_yellow_duration = self.calculate_new_durations(
                current_green_duration, current_yellow_duration, traffic_category)
            
            
            self.update_signal(intersection_name, lane, new_green_duration, new_yellow_duration, lane)

    @staticmethod
    def categorize_traffic(vehicles_detected):
        if vehicles_detected == 0:
            return "empty"
        elif vehicles_detected <= 10:
            return "low"
        elif vehicles_detected <= 23:
            return "medium"
        else:
            return "high"

    def calculate_new_durations(self, current_green_duration, current_yellow_duration, traffic_category):
        
        if traffic_category == "empty":
            new_green_duration = max(current_green_duration - 5, 10)  
        elif traffic_category == "low":
            new_green_duration = max(current_green_duration - 5, 10)  
        elif traffic_category == "medium":
            new_green_duration = current_green_duration 
        elif traffic_category == "high":
            new_green_duration = current_green_duration + 10  
        else:
            new_green_duration = current_green_duration  
        
        
        new_yellow_duration = current_yellow_duration
        
        return new_green_duration, new_yellow_duration

    def main_loop(self):
        
           
        yolo_output = detectTraffic(zip(videofiles,areaArr))  
            
        for intersection_name in self.signal_timings.keys():
               
            self.adjust_signals(intersection_name, yolo_output)
                
            self.print_signal_durations(intersection_name)
            
           

    def print_signal_durations(self, intersection_name):
      
        intersection = self.signal_timings[intersection_name]
        print(f"Intersection: {intersection_name}")
        for lane, data in intersection.items():
            green_duration = data['green_duration']
            yellow_duration = data['yellow_duration']
            current_lane = data['current_lane']
            print(f"{lane}: Green - {green_duration} seconds, Yellow - {yellow_duration} seconds, Current Lane: {current_lane}")




# def POINTS(event, x, y, flags, param):
#     if event == cv2.EVENT_MOUSEMOVE :  
#         colorsBGR = [x, y]
#         print(colorsBGR)


veichles = ['car','motorcycle','bus','truck','bicycle']
totalCount = []



def detectTraffic(videoarea):

    for video,area in videoarea:

        # print(video,area)

        cap = cv2.VideoCapture(video)
        model = yolov5.load('yolov5x.pt')
        model.conf = 0.01
        model.iou = 0.45  
        model.agnostic = False 
        model.multi_label = False  
        model.max_det = 1000 

        count=0

        roi = area

        st_time = time.time()
        end_time = st_time +3
        # print(st_time)

        ret, frame = cap.read()
        frame = cv2.resize(frame, (1020, 500))
        results = model(frame)
        
        for index, row in results.pandas().xyxy[0].iterrows():
            x1 = int(row['xmin'])
            y1 = int(row['ymin'])
            x2 = int(row['xmax'])
            y2 = int(row['ymax'])
            d = (row['name'])
            cx = int(x1 + x2) // 2
            cy = int(y1 + y2) // 2
            results = cv2.pointPolygonTest(np.array(area, np.int32), ((cx, cy)), False)
            # print(results)
            if results >= 0 and d.lower() in veichles:
                count += 1

        totalCount.append(count)   

    print(totalCount)
    return totalCount

          

if __name__ == '__main__':
    controller = TrafficSignalController()
    # Add an intersection with four lanes and initial signal durations
    initial_green_durations = [30, 30, 30, 30]

    controller.add_intersection("Intersection1", ["Lane1", "Lane2", "Lane3", "Lane4"],initial_green_durations)
    # Start the main control loop
    controller.aco_optimize()
