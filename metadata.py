import pydicom
import os
import os.path as osp
import pdb


#change to your location
DATA_FOLDER = '/Users/jacob/Documents/cs-capstone/data/3863/'
DATA_SAVE = '/Users/jacob/Documents/cs-capstone/data/anonymize/'

def get_file_list(dirName):
    # Get the list of all files in directory tree at given path
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(dirName):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]
    return listOfFiles

def get_all_patients(DATA_FOLDER):
    '''
    Function iterates through every dicom image and extracts the patient name information from each file
    and generates dict with counts for each patient
    '''
    patient_name_count = {}

    file_list = get_file_list(DATA_FOLDER)
    for full_f in file_list:
        #prevent macs from fucking up
        if('DS_Store' in full_f):
            continue
        dcm = pydicom.read_file(full_f)
        meta_inf = pydicom.filereader.dcmread(full_f)
        patient_name = meta_inf.PatientName
        if patient_name in patient_name_count:
            patient_name_count[patient_name] += 1
        else:
            patient_name_count[patient_name] = 0
        
    return patient_name_count

def bodypart_breakdown(DATA_FOLDER):
    '''
    Function creates dictionary with counts of all the different body parts
    '''

    body_part_count = {}

    file_list = get_file_list(DATA_FOLDER)
    for full_f in file_list:
        #prevent macs from fucking up
        if('DS_Store' in full_f):
            continue
        dcm = pydicom.read_file(full_f)
        meta_inf = pydicom.filereader.dcmread(full_f)
        body_part = meta_inf.BodyPartExamined
        if body_part in body_part_count:
            body_part_count[body_part] += 1
        else:
            body_part_count[body_part] = 0
        
    return body_part_count


def anonymize_data(DATA_FOLDER,DATA_SAVE_PATH):
    '''
    Iterates through every dicom image and removes any private or revealing information
    DATA_SAVE_PATH: where to output anonymized dicom files
    '''

    if not os.path.exists(DATA_SAVE_PATH):
        os.makedirs(DATA_SAVE_PATH)
    file_list = get_file_list(DATA_FOLDER)
    for full_f in file_list:
        #prevent macs from fucking up
        if('DS_Store' in full_f):
            continue
        
        dcm = pydicom.read_file(full_f)
        meta_inf = pydicom.filereader.dcmread(full_f)

        tags_to_anonymize = ['PatientID','PatientBirthDate','PatientName', 'PatientAddress']

        for tag in tags_to_anonymize:
            if(tag in meta_inf):
                meta_inf.data_element(tag).value = '********'

        output_filename = DATA_SAVE_PATH+full_f.split('/')[-1]
        meta_inf.save_as(output_filename)


        
        





# d = get_all_patients(DATA_FOLDER)
# body_parts = bodypart_breakdown(DATA_FOLDER)
# anonymize_data(DATA_FOLDER,DATA_SAVE)
# pdb.set_trace()


