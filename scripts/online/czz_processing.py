# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 10:45:00 2023

@author: Vishakh Pillai - ATTD Yield Prediction & Modeling
- Based roughly on script provided from kfoley

"""

import os
import shutil
import glob
from hex import HEX_Model as hex_model
from hex import resize_images, resize, pad
from dateutil import tz
from datetime import datetime, timedelta
import csv
import time
import pandas as pd
from datetime import datetime
import logging
import logging.handlers

# debug flag
debugging = False

DOC_PATH = r'\\atdfile1\DETD\Metro\DOC\DOC_Input'
CZZ_OPERS = []
CZZ_XML_SOURCE_PATH = r"C:\HEX_ADC\CZZ\MAU_INPUT\XML\source" 
CZZ_XML_PROCESSING_PATH = r'C:\HEX_ADC\CZZ\XML\processing'
CZZ_XML_PROCESSED_PATH = r'C:\HEX_ADC\CZZ\XML\processed'
CZZ_LOG_FILE = r'C:\HEX_ADC\CZZ\Log\LogFile.txt'
CZZ_ERROR_LOG_FILE = r'C:\HEX_ADC\CZZ\Log\ErrorLog.txt'
CZZ_CSV_ARCHIVE = r'C:\HEX_ADC\CZZ\CSV\archive'
CZZ_CSV_PROCESSED = r'C:\HEX_ADC\CZZ\CSV\processed'
CZZ_CSV_PROCESSING = r'C:\HEX_ADC\CZZ\CSV\processing'
CZZ_CSV_OUTPUT = r'C:\HEX_ADC\CZZ\panel_output'
CZZ_PANEL_TEMP = r'C:\HEX_ADC\CZZ\PANEL_TEMP'

def get_lot(fname):
    """return the lot out of an xml filename"""
    fbase = os.path.basename(fname)
    ystart = fbase.find("_Y")+1
    opstart = fbase.find("_",ystart)+1
    opend = fbase.find("_", opstart)
    lotname = fbase[ystart:opstart-1]
    opname = fbase[opstart:opend]
    return lotname, opname

def move(src, dst):
    """Move file at src path to folder at dst path. If src file exists in dst folder it will be deleted before the move. Returns move_to_path."""
    try:
        logger.info('Moving {} to {}.'.format(src, dst))
        move_to_path = os.path.join(dst, os.path.basename(src))
        if os.path.exists(move_to_path):
            logger.info('Deleting Existing File: {} Prior to Move.'.format(move_to_path))
            os.remove(move_to_path)
        shutil.move(src, dst)
        logger.info('Done Moving {} to {}.'.format(src, dst))
        return move_to_path
    except Exception as e:
        logger.exception('Failed to Move File: {} to {}!'.format(src, dst))
        EXCEPTIONS.append('Move File Exception: ' + str(e))

def setup_logging():
    """Enable logging, set date format, logging level, log file path, and file mode."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    handler = logging.handlers.RotatingFileHandler(CZZ_LOG_FILE, mode='a', maxBytes=10485760, backupCount=10)
    handler.setLevel(logging.INFO)
    
    error_handler = logging.handlers.RotatingFileHandler(CZZ_ERROR_LOG_FILE, mode='a', maxBytes=10485760, backupCount=10)
    error_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    logger.addHandler(error_handler)

    logger.info('Configured and started logging system.')
    return logger

def get_xml_simple(xml_path):
    """This function will just return the file name without an extension.
    it must be an XML file. This can be used to trigger ADC"""
    try:
        logger.info('Checking for XML dropped in source folder')
        file_list=glob.glob(str(xml_path)+"\\*.xml")
        return file_list
    except:
        logger.exception('Failed to Get Latest MAU XML Lots!')

def create_temp(panel_csv, resize=1):
    df = pd.read_csv(panel_csv)
    temp_panel = os.path.join(CZZ_PANEL_TEMP, panel_csv.split('_Defect_List')[0])
    temp_img_folder = os.path.join(temp_panel,'mau_temp')
    temp_output = os.path.join(temp_panel, 'output')
    if not os.path.exists(temp_panel):
        os.makedirs(temp_panel)
    if not os.path.exists(temp_img_folder):
        os.makedirs(temp_img_folder)
    if not os.path.exists(temp_output):
        os.makedirs(temp_output)
    df['Full_MAU_IMAGE_PATH'] = df['MAU_IMAGE_PATH'].astype(str) + df['MAU_IMAGE'].astype(str)
    if resize:
        resize_images(input_full_path=df['Full_MAU_IMAGE_PATH'].to_list(), \
                    output_dir=temp_img_folder)
    else:
        for img in df['Full_MAU_IMAGE_PATH'].to_list():
            shutil.copy(img, temp_img_folder)
    return temp_panel, temp_img_folder, temp_output

def run_ADC_processing(xml_file):
    """Run ADC post processing on one MAU XML file."""
    
    #%% XML parsing chunk
    # get the lot, zip file location, and then extract images
    #
    # TODO: need to find original xml location to add to define as xmlfile to parse
    #
    # example xml for AOI202
    #xmlfile = "2021-08-17_15-11-43.4606_Y131T02E_8828_BAI20002_C_20210817151156.XML"
    
    #XML_SOURCE_PATH = r'\\atdfile2\SPTD_ADC\MAU_INPUT\xml\source' # this is where we should look for xml
    
    #xml_file = "2021-09-09_13-15-55.2738_Y107X020_8030_MAU202.xml"
    
    #lot_name = os.path.split(xml_file)[1]
    lot_run, operation_run = get_lot(xml_file)
    lot = lot_run
    
    doc_dff_location = DOC_PATH + "\\" + operation_run + "\\" + lot_run
    doc_in_list = glob.glob(DOC_PATH + "\\" + operation_run + "\\" + lot_run + "\\*csv")

    if operation_run not in CZZ_OPERS:
        logger.error(f'The operation {operation_run} for {lot_run} is not part of expected list of CZZ operations.')
        raise(Exception(f'{lot_run} cannot be processed, as {operation_run} is not an expected CZZ operation.'))

    logger.info('====================================================================================================')
    logger.info('Started Processing: %s', xml_file)
    
    panel_count = len(doc_in_list)
    logger.info('Lot number %s', str(lot_run))
    logger.info('Operation number %s', str(operation_run))
    logger.info('Looking in folder %s', str(doc_dff_location))
    logger.info('File list %s', str(doc_in_list))
    logger.info('Found %s panel CSVs', str(panel_count))
    logger.info('Copying CSVs to local folder for processing')
    move(xml_file, CZZ_XML_PROCESSING_PATH)
    
    merged_files = []

    
    for panel_file in doc_in_list:
        
        panel_start_time = (datetime.now(tz.tzlocal())).strftime(r'%m/%d/%Y %H:%M:%S')

        # Copy panel file and create temp folders and preprocess images
        shutil.copy(panel_file,CZZ_CSV_PROCESSING)
        t_panel_folder, t_img_folder, t_output = create_temp(panel_file, resize=1)
        
        # Run ADC and get output
        logger.info(f'Running ADC on {panel_file}')
        model = hex_model(model_path=os.listdir(os.path.join('models','CZZ'))[0], output_folder=t_output, \
                        image_folder= t_img_folder)
        output_df = model.pipeline()

        panel_end_time = (datetime.now(tz.tzlocal())).strftime(r'%m/%d/%Y %H:%M:%S')

        panel_df = pd.read_csv(panel_file)
        merged_dff = pd.merge(panel_df, output_df, on='MAU_IMAGE', how='left')
        merged_dff['ADC_CLASS'] = np.where(
            (merged_dff['ADC_CLASS']=='TRANS_FM') & ((merged_dff['DEFECT_TYPE'] == 'OPEN PLATING ERROR') |
            (merged_dff['DEFECT_TYPE'] == 'MISSING CONDUCTOR')),
            'OPEN', merged_dff['ADC_CLASS'])
        merged_dff['CLASSIFICATION']=merged_dff['ADC_CLASS']
        merged_dff['ADC_COEFF'] = merged_dff['CONF']
        merged_dff['ADC_START_DATE_TIME'] = panel_start_time
        merged_dff['ADC_END_DATE_TIME'] = panel_end_time
        merged_dff['ADC_SCRIPT_VERSION'] = '1.0.0.0.0'
        merged_dff['ADC_SCRIPT_STATUS'] = "SUCCESS"
        merged_dff.drop(merged_dff.columns[-1],axis=1, inplace=True)
        merged_dff['CLASSIFICATION_METHOD'] = 'ADC'   # assigning ADC to class method
        merged_dff['DISPOSITION'] = 'PASS' #pass dispo
        dff_output_path = os.path.join(CZZ_CSV_OUTPUT, os.path.basename(panel_file))
        merged_dff.to_csv(dff_output_path, index=False)
        merged_files.append(dff_output_path)

        logger.info('Clean out temp folder structure')
        shutil.rmtree(t_img_folder)
        shutil.rmtree(t_output)
        shutil.rmtree(t_panel_folder)
        
    logger.info('Moving CSVs back to DOC DFF location')
    for new_file in merged_files:
        move(new_file, doc_dff_location)

    logger.info('Moving CSVs from processing to processed')
    for processed_files in os.listdir(CZZ_CSV_PROCESSING):
        move(CZZ_CSV_PROCESSING + "\\" + processed_files, CZZ_CSV_PROCESSED)

    logger.info('Moving ADC output files to archive')
    for output_file in os.listdir(CZZ_CSV_OUTPUT):
        move(output_file, CZZ_CSV_ARCHIVE)

    logger.info('Moving XML to processed folder')
    move(xml_file, CZZ_XML_PROCESSED_PATH)
    
if __name__ == "__main__":
    """Main Function that runs ADC Flow."""
    logger = setup_logging()

    #use a few lines of code to set up the machine learning:
    #featdf=pd.read_csv(ADC_LIBRARY)
    #featdf=featdf.drop(featdf[featdf['defect size']<4].index)
    
    #use the ii terms to test the loop
    ii=0
    
    while True:
        #while ii<100:
        #lot_list = get_latest_lots(XML_SOURCE_PATH)
        lot_list = get_xml_simple(CZZ_XML_SOURCE_PATH)
        
        if not lot_list:
            if ii >=5:
                lot_list = get_xml_simple(CZZ_XML_SOURCE_PATH)
                logger.info('Checking for delayed lots')
                ii=0
            
        if lot_list:
            for lot_path in lot_list:
                EXCEPTIONS = []
                try:
                    #ii+=1000
                    panels=run_ADC_processing(lot_path)
                except Exception as e:
                    logger.exception('Failed to process AOI XML File: %s!'%(lot_path))
                    EXCEPTIONS.append('Run Processing Exception: ' + str(e))
        else:
            #time.sleep(10)
            time.sleep(20)
            ii+=1
