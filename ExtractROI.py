import utils


def GeneratePatchImages(mask_dir, img_dir, img_dir_to_store):
    print(img_dir_to_store)
    fd = utils.Filedirectoryamagement(mask_dir, img_dir)
    fd.LoadFiles()
    fd.ExtractROI(img_dir_to_store)

if __name__ == '__main__':
    GeneratePatchImages("/Volumes/Volume/Histo_image_For_Dr_Kunder/Done","/Volumes/Volume/Histo_image_For_Dr_Kunder/Done","/Users/okyazeminaga/Desktop/ROI/")