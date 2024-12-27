import sys
import os

import argparse
from pathlib import Path

from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.image import InterpolationMethod
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import RecordableTypeId, StreamId

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image



def image_config_example(config):
    print(f"device_type {config.device_type}")
    print(f"device_version {config.device_version}")
    print(f"device_serial {config.device_serial}")
    print(f"sensor_serial {config.sensor_serial}")
    print(f"nominal_rate_hz {config.nominal_rate_hz}")
    print(f"image_width {config.image_width}")
    print(f"image_height {config.image_height}")
    print(f"pixel_format {config.pixel_format}")
    print(f"gamma_factor {config.gamma_factor}")



def showBeforeAndAfterUndistortion(image_array, rectified_array, sensor_name):

    # visualize input and results
    # plt.figure()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(image_array, cmap="gray", vmin=0, vmax=255)
    axes[0].title.set_text(f"sensor image ({sensor_name})")
    axes[0].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    axes[1].imshow(rectified_array, cmap="gray", vmin=0, vmax=255)
    axes[1].title.set_text(f"undistorted image ({sensor_name})")
    axes[1].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    plt.show()


def undistortAndSaveImages(filename, output_dir, sensor_name):
    # Create a data provider
    vrsfile = filename
    print(f"Creating data provider from {vrsfile}")
    provider = data_provider.create_vrs_data_provider(vrsfile.as_posix())
    if not provider:
        print("Invalid vrs data provider")

    # sensor_name = "camera-rgb"
    sensor_stream_id = provider.get_stream_id_from_label(sensor_name)

    # Retrieve and show statistics
    num_data = provider.get_num_data(sensor_stream_id)
    print("I have found", num_data, " images from ", sensor_name)
    config = provider.get_image_configuration(provider.get_stream_id_from_label(sensor_name))
    image_config_example(config)

    sensor_stream_id = provider.get_stream_id_from_label(sensor_name)
    # input: retrieve image distortion
    device_calib = provider.get_device_calibration()
    src_calib = device_calib.get_camera_calib(sensor_name)

    # create output calibration: a linear model of image size 512x512 and focal length 150
    # Invisible pixels are shown as black.
    dst_calib = calibration.get_linear_camera_calibration(512, 512, 150, sensor_name)


    for image_idx in range(0, num_data, 5):
    # for image_idx in range(100, 150, 5):
        image_data = provider.get_image_data_by_index(sensor_stream_id, image_idx)
        image_array = image_data[0].to_numpy_array()
        # distort image
        rectified_array = calibration.distort_by_calibration(image_array, dst_calib, src_calib, InterpolationMethod.BILINEAR)
        # showBeforeAndAfterUndistortion(image_array, rectified_array, sensor_name)
        image_PIL = Image.fromarray(rectified_array)
        image_name = output_dir / f"{sensor_name}_{image_idx:05}.png"
        image_PIL.save(image_name)
        print("Image saved to", image_name)



def main():
    parser = argparse.ArgumentParser(description="Extracting images from vrs")
    parser.add_argument(
    "--file", 
    type=Path,
    required=True, 
    help="File to parse"
    )

    parser.add_argument(
        "--sensor_name",
        type=str,
        choices=("camera-rgb", "camera-slam-right", "camera-slam-left"),
        required=True,
        help="camera from which to extract images"
    )
    parser.add_argument("--output_dir", type=Path, help="output directory", required= True)
    args = parser.parse_args()
    output_dir_sensor_name = args.output_dir / args.sensor_name
    if output_dir_sensor_name.exists():
        print("Image folder exists. Skipping extraction.")
        return
    output_dir_sensor_name.mkdir()
    undistortAndSaveImages(args.file, output_dir_sensor_name, args.sensor_name)



if __name__=="__main__":
    main()

