#############################
#  Author : Naga Anjaneyulu #
#         IU Research       #
############################
import math
import os
import csv
import copy
import random
import numpy as np
import pandas as pd
from manage import db
import matplotlib.pyplot as plt
from app import gb_job_skill_rel, gb_node_emb, gb_nodes, gb_job_skill_rel_tc, gb_rev_nodes, gb_ds_ground_truth, \
    gb_courses_desc, gb_path, gb_ass_matrix
from app import  gb_courses, gb_skill, gb_knowledge, gb_kc_rel
from model.models import Job, User, AssesmentData, Question, Response, UserSatisfaction, GroundTruth
from sklearn.metrics.pairwise import cosine_similarity


"@ function for building assesment object"
def build_assesment_data(data):
    job_data = data.get("job", None)
    user = data.get("user", None)
    saved_user = User.query.filter_by(id=user.get("id", None)).first()
    job = Job(saved_user.get_id(), job_data.get("category1"), job_data.get("category2"), job_data.get("category3"))
    job.user = saved_user
    db.session.add(job)
    db.session.commit()
    saved_job = Job.query.filter_by(user_id=job.get_user_id()).first()
    ass_data = AssesmentData(saved_user.get_id(), saved_job.id, saved_job.category1, saved_job.category2, saved_job.category3)
    return ass_data

"@ function for parse assesment object"
def parse_assesment_data(data):
    ass_data = AssesmentData(data.get("user_id",None), data.get("job_id",None),
                             data.get("category1",None), data.get("category2",None) ,data.get("category3",None))
    ass_data.set_count(data.get("count",0))
    ass_data.set_response(data.get("response","NA"))
    ass_data.set_pseudo(data.get("pseudo",{}))
    return ass_data



"@function for generating pseudo recommendation"
def generate_pseudo_recomm(ass_matrix,category,pseudo):
    ass_matrix['sum'] = ass_matrix[list(ass_matrix.columns)].sum(axis=1)
    ass_matrix = ass_matrix.sort_values(by='sum', axis=0, ascending=False, inplace=False, kind='quicksort')
    count = 0
    for index, rows in ass_matrix.iterrows():
        if index not in pseudo.keys() and index in gb_rev_nodes.keys() and gb_rev_nodes[index] in gb_courses.keys() and gb_rev_nodes[index]  in  gb_courses_desc.keys() and (count < 10):
            pseudo[index] = { "courseid" : gb_rev_nodes[index] , "coursename" : gb_courses[gb_rev_nodes[index]]  ,"coursedesc" : gb_courses_desc[gb_rev_nodes[index]] , "response" : "K" , "category": category,"rank" : count}
            count += 1
    return pseudo


"@ function for starting assesment test"
def start_assesment(data):
    pseudo_recom = {}
    ass_data = build_assesment_data(data)
    path = "/home/nakopa/crs-project/crs/resources/data/"
    ass_matrix = pd.read_csv(os.path.join(path, gb_ass_matrix[ass_data.get_category1()]))
    generate_pseudo_recomm(ass_matrix,ass_data.get_category1(),pseudo_recom)
    ass_matrix = pd.read_csv(os.path.join(path, gb_ass_matrix[ass_data.get_category2()]))
    generate_pseudo_recomm(ass_matrix, ass_data.get_category2(), pseudo_recom)
    ass_matrix = pd.read_csv(os.path.join(path, gb_ass_matrix[ass_data.get_category1()]))
    generate_pseudo_recomm(ass_matrix, ass_data.get_category3(), pseudo_recom)
    ass_data.set_pseudo(pseudo_recom)

    return ass_data


"@ function for next question generation"
def next_question(data):
    ass_data = parse_assesment_data(data)
    recommend = 0
    if ass_data.response != "Yes":
        recommend = 1
    response = Response(ass_data.get_user_id(),ass_data.get_course_id(),ass_data.get_course_name(),ass_data.get_response(),recommend)
    db.session.add(response)
    db.session.commit()
    ass_data.used_courses += [ass_data.course_id]
    if(recommend == 1):
        ass_data.recommend += [ass_data.course_id]
    generate_question(ass_data)
    return ass_data

"@ function for user satisfaction"
def user_satisfaction(data):
    ass_data = parse_assesment_data(data)
    refine_recommend = {}
    not_recommend = {}
    gt ={}
    for key,value in ass_data.pseudo.items():
        if value["response"] == "K":
            response = Response(ass_data.get_user_id(), value["courseid"], value["coursename"],
                                value["response"],0,value["category"])
            db.session.add(response)
            db.session.commit()
            not_recommend[key] = value
        else:
            response = Response(ass_data.get_user_id(), value["courseid"], value["coursename"],
                                value["response"], 1, value["category"])
            db.session.add(response)
            db.session.commit()
            refine_recommend[key] = value
        gt[key] = value
        gt[key]["choice"] =  "SA"

    gt = dict(sorted(gt.items(), key=lambda kv: kv[1]["rank"]))
    not_recommend = dict(sorted(not_recommend.items(), key=lambda kv: kv[1]["rank"]))
    refine_recommend = dict(sorted(refine_recommend.items(), key=lambda kv: kv[1]["rank"]))
    sat_data = UserSatisfaction(ass_data.user_id)
    sat_data.set_gt(gt)
    sat_data.set_not_recommend(not_recommend)
    sat_data.set_pseudo_recommend(ass_data.get_pesudo())
    sat_data.set_refine_recommend(refine_recommend)
    sat_data.set_count(ass_data.get_count())
    sat_data.set_category1(ass_data.get_category1())
    sat_data.set_category2(ass_data.get_category2())
    sat_data.set_category3(ass_data.get_category3())
    sat_data.set_category(ass_data.get_category1())
    return sat_data

"@ function for parsing user satisfaction data"
def parse_user_sat_data(data):
    sat_data = UserSatisfaction(data.get("user_id", None))
    sat_data.set_refine_quality(data.get("refine_quality", "No"))
    sat_data.set_category(data.get("category", "NA"))
    sat_data.set_category1(data.get("category1", "NA"))
    sat_data.set_category2(data.get("category2", "NA"))
    sat_data.set_category3(data.get("category3", "NA"))
    sat_data.set_count(data.get("count", 0))
    sat_data.set_pseudo_recommend(data.get("pseudo_recommend", {}))
    sat_data.set_refine_recommend(data.get("refine_recommend", {}))
    sat_data.set_not_recommend(data.get("not_recommend", {}))
    sat_data.set_gt(data.get("gt", {}))
    return sat_data

"@ function for user satisfaction"
def recommend_courses(data):
    sat_data = parse_user_sat_data(data)
    db.session.add(sat_data)
    db.session.commit()
    return sat_data

"@ function for generating report"
def generate_report(data):
    sat_data = parse_user_sat_data(data)
    path = os.path.join('/home/nakopa/crs-project/crs/results/', str(sat_data.get_user_id()))
    if sat_data.count == 1:
        os.mkdir(path)
    path = os.path.join(path, str(sat_data.count))
    os.mkdir(path)

    " Storing ground truth values "
    gt = sat_data.get_gt()
    with open(os.path.join(path,"ground_truth.csv"), 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        rows = []
        rows.append(["UserID", "CourseID", "Course Name",  "Choice" ,"Ground Truth"])
        for key, value in gt.items():
            if value["category"] == sat_data.get_category():
                csv_row = []
                csv_row.append(sat_data.get_user_id())
                csv_row.append(value["courseid"])
                csv_row.append(value["coursename"])
                csv_row.append(value["choice"])
                gt_value = 0
                if str(gb_nodes[value["courseid"]]) in sat_data.get_refine_recommend().keys():
                    if(value["choice"] == "SA" ):
                        gt_value = 2
                    elif (value["choice"] == "A" ):
                        gt_value = 1
                    elif (value["choice"] == "D"):
                        gt_value = -1
                    elif (value["choice"] == "SD"):
                        gt_value = -2
                    else :
                        gt_value = 0
                if str(gb_nodes[value["courseid"]]) in sat_data.get_not_recommend().keys():
                    if (value["choice"] == "SA"):
                        gt_value = 2
                    elif (value["choice"] == "A"):
                        gt_value = 1
                    elif (value["choice"] == "D"):
                        gt_value = -1
                    elif (value["choice"] == "SD"):
                        gt_value = -2
                    else:
                        gt_value = 0
                csv_row.append(gt_value)
                rows.append(csv_row)
                gt_obj = GroundTruth(sat_data.get_user_id(),value["courseid"],value["coursename"],value["choice"],gt_value)
                db.session.add(gt_obj)
                db.session.commit()
        writer.writerows(rows)
        csv_file.close()
    " Storing user profile "
    if sat_data.count == 1 :
        saved_user = User.query.filter_by(id=sat_data.get_user_id()).first()
        job = Job.query.filter_by(user_id=saved_user.get_id()).first()
        with open(os.path.join(path, "user_profile.csv"), 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            rows = []
            rows.append(["UserID", "Username", "Job Category 1", "Job Category 2", "Job Category 3"])
            csv_row = []
            csv_row.append(saved_user.get_id())
            csv_row.append(saved_user.get_username())
            csv_row.append(job.get_category1())
            csv_row.append(job.get_category2())
            csv_row.append(job.get_category3())
            rows.append(csv_row)
            writer.writerows(rows)
            csv_file.close()

    " Storing user assessment responses "
    if sat_data.count == 1 :
        with open(os.path.join(path, "user_assessment.csv"), 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            rows = []
            rows.append(["UserID", "CourseID", "Course Name", "Response", "Recommendation","Category"])
            response_list = Response.query.filter_by(user_id = sat_data.get_user_id()).all()
            print(len(response_list))
            for response in response_list:
                csv_row = []
                csv_row.append(response.user_id)
                csv_row.append(response.course_id)
                csv_row.append(response.course_name)
                csv_row.append(response.response)
                csv_row.append(response.recommend)
                csv_row.append(response.category)
                rows.append(csv_row)
            writer.writerows(rows)
            csv_file.close()

    " Storing user satisfaction responses "
    if sat_data.count == 3 :
        with open(os.path.join(path, "user_satisfaction.csv"), 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            rows = []
            rows.append(["UserID", "Refine Quality", "Category"])
            sat_obj_list = UserSatisfaction.query.filter_by(user_id=sat_data.get_user_id()).all()
            for sat_obj in sat_obj_list :
                csv_row = []
                csv_row.append(sat_obj.user_id)
                csv_row.append(sat_obj.refine_quality)
                csv_row.append(sat_obj.category)
                rows.append(csv_row)
            writer.writerows(rows)
            csv_file.close()

    " Storing refined recommendation "
    with open(os.path.join(path, "actual_recommendation.csv"), 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        rows = []
        rows.append(["CourseID", "CourseName", "Category" ,"Recommendation"])
        for key, value in sat_data.get_refine_recommend().items():
            if value["category"] == sat_data.category :
                csv_row = []
                csv_row.append(value["courseid"])
                csv_row.append(value["coursename"])
                csv_row.append(value["category"])
                csv_row.append(1)
                rows.append(csv_row)
        writer.writerows(rows)
        csv_file.close()

    " Storing not recommended courses"
    with open(os.path.join(path, "not_recommended.csv"), 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        rows = []
        rows.append(["CourseID", "CourseName", "Category" ,"Recommendation" ])
        for key, value in sat_data.get_not_recommend().items():
            if value["category"] == sat_data.category:
                csv_row = []
                csv_row.append(value["courseid"])
                csv_row.append(value["coursename"])
                csv_row.append(value["category"])
                csv_row.append(0)
                rows.append(csv_row)
        writer.writerows(rows)
        csv_file.close()

    sat_data.set_count(sat_data.get_count() + 1)
    if(sat_data.count == 1):
        sat_data.set_category(sat_data.get_category1())
    elif sat_data.count == 2 :
        sat_data.set_category(sat_data.get_category2())
    else :
        sat_data.set_category(sat_data.get_category3())
    sat_data.set_refine_quality("")
    return sat_data

