import json
import glom
import os

def add_in_list(path_to_list, iterate_over, complete_json, to_append):
    to_be_iterate = []
    secondary_path = ""
    primary_path = ""
    for i in range(0,len(path_to_list)):

        if iterate_over >= i:
            to_be_iterate = complete_json[path_to_list[i]]
            if primary_path == "":
                primary_path += path_to_list[i]

            else:
                primary_path += "." + primary_path

        else:
            if secondary_path == "":
                secondary_path += path_to_list[i]

            else:
                secondary_path += "." + secondary_path

    temp_list = []   

    for i in range(0,len(to_be_iterate)):
        final_list = glom.glom(to_be_iterate[i], secondary_path)
        final_list.append(to_append)
        glom.assign(to_be_iterate[i], secondary_path, final_list)
    
    glom.assign(complete_json, primary_path, to_be_iterate)  
    return complete_json   

def replace_in_algo(algo_type, field_to_replace, replace_with,complete_json):
    all_cameras = complete_json["cameras"]
    
    for i in range(0, len(all_cameras)):
        algos = all_cameras[i]["algorithms"]

        for j in range(0,len(algos)):   
            if algos[j]["algo"] == algo_type:
                algos[j][field_to_replace] = replace_with

        all_cameras[i]["algorithms"] = algos

    complete_json["cameras"] = all_cameras

    return complete_json

def add_in_list2(path_to_list, iterate_over, complete_json, to_append):
    to_be_iterate = []
    secondary_path = ""
    primary_path = ""
    for i in range(0,len(path_to_list)):

        if iterate_over >= i:
            to_be_iterate = complete_json[path_to_list[i]]
            if primary_path == "":
                primary_path += path_to_list[i]

            else:
                primary_path += "." + primary_path

        else:
            if secondary_path == "":
                secondary_path += path_to_list[i]

            else:
                secondary_path += "." + secondary_path

    temp_list = []   

    for i in range(0,len(to_be_iterate)):
        final_list = glom.glom(to_be_iterate[i], secondary_path)
        final_list.append(to_append)

        glom.assign(to_be_iterate[i], secondary_path, final_list)
    
    glom.assign(complete_json, primary_path, to_be_iterate)  
    return complete_json
    
def add_queue_algo(complete_json):
    all_cams = complete_json["cameras"]
    all_queue = []
    sample_algo = {
                    "algo": "QUEING",
                    "threshold_time": 20,
                    "roi_conf": [
                        {
                            "ROI": [
                                {
                                    "x": 0.606,
                                    "y": 0.275
                                },
                                {
                                    "x": 0.791,
                                    "y": 0.537
                                },
                                {
                                    "x": 0.006,
                                    "y": 0.798
                                },
                                {
                                    "x": 0.003,
                                    "y": 0.454
                                }
                            ],
                            "unique_roi_id": "DU4"
                        }
                    ]
                }
    for i in range(0,len(all_cams)):
        sample_algo = {
                    "algo": "QUEING",
                    "threshold_time": 20,
                    "roi_conf": []
                }
        for j in range(0, len(all_cams[i]["algorithms"])):

            if all_cams[i]["algorithms"][j]["algo"] == "FUELLING":

                for k in range(0,len(all_cams[i]["algorithms"][j]["roi_conf"])):
                    queue_roi = all_cams[i]["algorithms"][j]["roi_conf"][k]["roi_FSM"]
                    du_name = all_cams[i]["algorithms"][j]["roi_conf"][k]["unique_roi_id"]

                    sample_algo["roi_conf"].append({"ROI": queue_roi, "unique_roi_id": du_name})
        
                all_cams[i]["algorithms"].append(sample_algo)

    complete_json["cameras"] = all_cams
    return complete_json

def add_uniform_algo(complete_json):
    all_cams = complete_json["cameras"]
    all_queue = []
    sample_algo = {
                    "algo": "UNIFORM_CHECK",
                    "roiconfig": [
                        {
                            "frame_count_threshold_for_violation": 20,
                            "unique_roi_id": "DU15",
                            "roi": [
                                { "x": 0.612, "y": 0.364 },
                                { "x": 0.854, "y": 0.386 },
                                { "x": 0.926, "y": 0.652 },
                                { "x": 0.633, "y": 0.620 }
                            ]
                        }
                    ]
                }
    for i in range(0,len(all_cams)):
        sample_algo = {
                    "algo": "UNIFORM_CHECK",
                    "roiconfig": []
                }
        for j in range(0, len(all_cams[i]["algorithms"])):

            if all_cams[i]["algorithms"][j]["algo"] == "FUELLING":

                for k in range(0,len(all_cams[i]["algorithms"][j]["roi_conf"])):
                    queue_roi = all_cams[i]["algorithms"][j]["roi_conf"][k]["roi_FSM"]
                    du_name = all_cams[i]["algorithms"][j]["roi_conf"][k]["unique_roi_id"]

                    sample_algo["roiconfig"].append({"frame_count_threshold_for_violation": 30,"unique_roi_id": du_name,"roi": queue_roi})
        
                all_cams[i]["algorithms"].append(sample_algo)

    complete_json["cameras"] = all_cams
    print(queue_roi)
    return complete_json

def roi_extract_algo(complete_json):
    all_cams = complete_json["cameras"]
    print(len(all_cams))
    all_queue = []
    for i in range(0,len(all_cams)):
        for j in range(0, len(all_cams[i]["algorithms"])):
            if all_cams[i]["algorithms"][j]["algo"] == "SPILLAGE":
                for k in range(0,len(all_cams[i]["algorithms"][j]["roi_conf"])):
                    queue_roi = all_cams[i]["algorithms"][j]["roi_conf"][k]
                    du_name = all_cams[i]["algorithms"][j]["roi_conf"][k]["unique_roi_id"]
                    print(queue_roi)
                    print("ROI ID:", du_name)

                    

    # complete_json["cameras"] = all_cams
                
    # return complete_json

if __name__ == "__main__":

    all_files = os.listdir('/home/ninad/Music/roi_crop_common/cfgs')
    print(all_files)

    for k in range(0, len(all_files)):
        if all_files[k] == "changed_cfg" or all_files[k] == "modify_cfg.py" or all_files[k] == "Violations.xlsx":
            continue
        with open(all_files[k], 'r') as openfile:
            current_file = json.load(openfile)

        #print(current_file)
        '''
        a = {
                        "algo":"SOCIAL_DISTANCING",
                        "calib_info": {
                            "h_ratio":0.9,
                            "v_ratio":0.65,
                            "calibration":0.3
                        },
                        "alarm_cooldown_period":1000,
                        "sd_frames":3,
                        "sd_ratio":1,
                        "skip_frames":1,
                        "bg_sub":False,
                        "bg_sub_period":2000,
                        "_BG_SUB_THRESHOLD_COMMENT":"bg_sub_threshold between 0 - 100%",
                        "bg_sub_threshold":5,
                        "critical_alarm":False,
                        "_SD_CRITICAL_COMMENT_":"Set only if critical_alarm is true",
                        "critical_threshold":500
                    }

        b = {
                        "algo":"MASK_COMPLIANCE",
                        "start_score":2,
                        "face_score":5,
                        "mask_score":6,
                        "alarm_cooldown_period":3000,
                        "delete_threshold":100,
                        "solo_exemption":False
                    }
        '''
        try:
            # to_be_written = add_in_list(["cameras","algorithms"],0,current_file, a)
            # to_be_written = add_in_list(["cameras","algorithms"],0,to_be_written, b)
            # to_be_written = replace_in_algo("DECANTATION", "ROI_PERSON", [{ "x": 0.1, "y": 0.1 },{ "x": 0.98, "y": 0.1 },{ "x": 0.98, "y": 0.98 },{ "x": 0.1, "y": 0.98 }],to_be_written)
            # to_be_written = replace_in_algo("DECANTATION", "ROI_FIRE_EXTINGUISHER", [{ "x": 0.1, "y": 0.1 },{ "x": 0.98, "y": 0.1 },{ "x": 0.98, "y": 0.98 },{ "x": 0.1, "y": 0.98 }],to_be_written)
            # to_be_written = replace_in_algo("DECANTATION", "ROI_BARICADING", [{ "x": 0.1, "y": 0.1 },{ "x": 0.98, "y": 0.1 },{ "x": 0.98, "y": 0.98 },{ "x": 0.1, "y": 0.98 }],to_be_written)
            # to_be_written = replace_in_algo("DECANTATION", "ROI_TRUCK", [{ "x": 0.1, "y": 0.1 },{ "x": 0.98, "y": 0.1 },{ "x": 0.98, "y": 0.98 },{ "x": 0.1, "y": 0.98 }],to_be_written)
            # to_be_written = replace_in_algo("SOCIAL_DISTANCING", "alarm_cooldown_period", 30000, current_file)
            # to_be_written = replace_in_algo("UNIFORM_CHECK", "roiconfig.frame_count_threshold_for_violation", 40, current_file)
            # to_be_written = add_queue_algo(current_file)
            to_be_written = roi_extract_algo(current_file)
            # json_to_be_written = json.dumps(to_be_written, indent = 4)

            # with open("./changed_cfg/" + all_files[k], "w") as outfile:
            #     outfile.write(json_to_be_written)
        except Exception as e:
            print(e)
            print(all_files[k])

        