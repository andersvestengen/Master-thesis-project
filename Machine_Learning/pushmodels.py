#Short script to pull models that have finished training and are lying in server storage

from Training_Framework import FileSender

if __name__ == '__main__':
    fmg = FileSender()

    fmg.push_local_models()

    fmg.close()