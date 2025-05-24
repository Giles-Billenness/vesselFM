# How to Prepare the ImageCAS Dataset
From https://www.kaggle.com/datasets/xiaoweixumedicalai/imagecas/data?select=Coronary_Segmentation_deep_learning

1. Manually move the downloaded archive to ./Data/ImageCAS/
2. Unzip the main archive.
3. Rename any files ending in ".change2zip" to ".zip"
4. Unzip these new .zip files to the same directory.
5. Run the ImageCAS dataset converter script to:
   - Resize images to 256×256×slices
   - Convert *.nii.gz to *.npy
   - Create an 80/10/10 train/val/test split

On Windows, run in PowerShell:
```powershell
python .\vesselfm\d_real\convert_imagecas.py ^
>>     --source_dirs "C:\Users\giles\Github\vesselFM\data\ImageCAS\1-200" `
>>                   "C:\Users\giles\Github\vesselFM\data\ImageCAS\201-400" `
>>                   "C:\Users\giles\Github\vesselFM\data\ImageCAS\401-600" `
>>                   "C:\Users\giles\Github\vesselFM\data\ImageCAS\601-800" `
>>                   "C:\Users\giles\Github\vesselFM\data\ImageCAS\801-1000" `
>>     --output_base "C:\Users\giles\Github\vesselFM\data\d_real"
```

To visualize sample slices, add the `--plot_samples` flag.

