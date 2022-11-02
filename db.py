from os import name
import pymysql
from pymysql import cursors
import numpy as np


def get_cursor():
        connection = pymysql.connect(host='localhost',
                                user='root',
                                password='harsh@123',
                                database='cbir',
                                cursorclass=pymysql.cursors.DictCursor)
        cursor=connection.cursor()
        return connection,cursor


def add_img_feature_to_db(source,feature_dict):
        connection,cursor=get_cursor()
        glcm="$".join([str(i) for i in feature_dict["glcm"].tolist()])
        color_histogram="$".join([str(i) for i in feature_dict["color_histogram"].tolist()])
        st="INSERT INTO Features VALUES ('"+source+"','"+glcm+"','"+color_histogram+"');"
        cursor.execute(st)
        connection.commit()
        connection.close()


def get_db_features():
        connection,cursor=get_cursor()
        cursor.execute('''
        SELECT * FROM FEATURES;
        ''')
        result = cursor.fetchall()
        for i in range(len(result)):
                result[i]["glcm"]=np.asarray(result[i]["glcm"].split("$"),dtype=np.float32)
                result[i]["color_histogram"]=np.asarray(result[i]["color_histogram"].split("$"),dtype=np.float32)
        connection.close()
        return result


def Recreate_table():
        connection,cursor=get_cursor()
        delete_existing_table = "drop table if exists Features;"
        cursor.execute(delete_existing_table)


        create_new_table='''
        CREATE TABLE `Features` (
        `source` varchar(1024),
        `glcm` LONGTEXT,
        `color_histogram` LONGTEXT
        )
        '''
        cursor.execute(create_new_table)
        connection.commit()
        connection.close()


def showTable():
        connection,cursor=get_cursor()
        cursor.execute('''
        SELECT * FROM FEATURES;
        ''')
        result = cursor.fetchall()
        # print(result)
        connection.close()


def fillDummy():
        connection,cursor=get_cursor()
        source = "images/1.jpg"
        glcm="9.7624$9,76$6.987"
        color_histogram="7.83663$6.92737$9.564378"
        st="INSERT INTO Features VALUES ('"+source+"','"+glcm+"','"+color_histogram+"');"
        # print(st)
        cursor.execute(st)
        connection.commit()
        connection.close()


if __name__ == "__main__":
        Recreate_table()
        # fillDummy()
        # showTable()