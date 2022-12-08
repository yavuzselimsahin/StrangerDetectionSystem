import dataset_creator as dc
import face_training as ft
import face_recognition as fr


def main():
    while True:
        start = True
        q = input("\n [INFO] App is started, press 'q' to quit 'enter' to continue ")
        if q.lower() == "q":
            break

        while start:
            ans = input("\n Dou you want to add image to database for training and continue to app y/n ")
            if ans.lower() == "q":
                start = False

            elif ans.lower() == "y":
                dc.run()
                ft.run()
                fr.run()
                start = False

            elif ans.lower() == "n":
                print("\n Camera is opening please wait...")
                fr.run()
                start = False

    print("\n[INFO] App is closed")


if __name__ == "__main__":
    main()
