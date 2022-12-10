#Short script to pull models that have finished training and are lying in server storage

from Training_Framework import FileSender

if __name__ == '__main__':
    fmg = FileSender()

    fmg.get_remote_models()

    fmg.close()