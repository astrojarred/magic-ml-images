import shutil
import subprocess
from pathlib import Path

import dotenv
import magicpy.mars as mars
import numpy as np
import pandas as pd
import uproot


def process_calibrated(
    run_number: int,
    calibrated_m1_file: Path | str,
    calibrated_m2_file: Path | str,
    star_output_dir: Path | str,
    superstar_output_dir: Path | str | None = None,
    to_parquet: bool = True,
    parquet_file_prefix: str | None = None,
    max_event_number: int = 10000,
    overwrite_root: bool = True,
):
    # load in environment variables
    env = dict(dotenv.dotenv_values())

    if not env.get("MARSSYS"):
        raise ValueError("MARSSYS environment variable not set")
    if not env.get("ROOTSYS"):
        raise ValueError("ROOTSYS environment variable not set")

    if not superstar_output_dir:
        superstar_output_dir = star_output_dir

    if not isinstance(star_output_dir, Path):
        star_output_dir = Path(star_output_dir)
    if not isinstance(superstar_output_dir, Path):
        superstar_output_dir = Path(superstar_output_dir)

    if not star_output_dir.exists():
        star_output_dir.mkdir(parents=True)
    if not superstar_output_dir.exists():
        superstar_output_dir.mkdir(parents=True)

    if not isinstance(calibrated_m1_file, Path):
        calibrated_m1_file = Path(calibrated_m1_file)
    if not isinstance(calibrated_m2_file, Path):
        calibrated_m2_file = Path(calibrated_m2_file)

    # ensure the files exist
    if not calibrated_m1_file.exists():
        raise FileNotFoundError(f"Calibrated M1 file not found: {calibrated_m1_file}")
    if not calibrated_m2_file.exists():
        raise FileNotFoundError(f"Calibrated M2 file not found: {calibrated_m2_file}")

    starm1 = mars.Star(
        input_files=[calibrated_m1_file],
        analysis_path=star_output_dir,
        output_dir=star_output_dir,
        nickname=f"StarM1-DHBW-gamma-{run_number}",
        telescope_number=1,
        is_mc=True,
    )

    starm2 = mars.Star(
        input_files=[calibrated_m2_file],
        analysis_path=star_output_dir,
        output_dir=star_output_dir,
        nickname=f"StarM2-DHBW-gamma-{run_number}",
        telescope_number=2,
        is_mc=True,
    )

    resm1 = starm1.run(
        file=calibrated_m1_file,
        save_all=True,
        overwrite=overwrite_root,
    )

    resm2 = starm2.run(
        file=calibrated_m2_file,
        save_all=True,
        overwrite=overwrite_root,
    )

    if resm1.get("return_code") != 0:
        raise ValueError(f"Error running M1 star: \n{resm1['error']}")
    if resm2.get("return_code") != 0:
        raise ValueError(f"Error running M2 star: \n{resm2['error']}")

    star_m1_filepath = resm1["star_file_path"]
    star_m2_filepath = resm2["star_file_path"]

    star_m1_regex = star_m1_filepath.parent / f"{star_m1_filepath.stem}*.root"
    star_m2_regex = star_m2_filepath.parent / f"{star_m2_filepath.stem}*.root"

    superstar = mars.SuperStar(
        input_files=[star_m1_regex, star_m2_regex],
        analysis_path=superstar_output_dir,
        output_dir=superstar_output_dir,
        nickname=f"SuperStar-DHBW-gamma-{run_number}",
        is_mc=True,
    )

    ssres = superstar.run(
        m1_file=star_m1_regex,
        m2_file=star_m2_regex,
        is_mc=True,
        log_file=superstar_output_dir / f"SuperStar-DHBW-gamma-{run_number}.log",
        overwrite=overwrite_root,
    )

    if ssres.get("return_code") != 0:
        raise ValueError(f"Error running SuperStar: \n{ssres['error']}")

    superstar_filepath = ssres["superstar_file"]

    # run the writeImages2uproot macro
    eof = f'root -b -l -q {env["MARSSYS"]}/macros/writeImages2uproot.C("{superstar_filepath.name}")'

    process = subprocess.Popen(
        eof.split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=superstar_output_dir,
        env=superstar.env | env,
    )

    output, error = process.communicate(timeout=300)

    if process.returncode != 0:
        raise ValueError(f"Error running writeImages2uproot.C: {error}")

    res = {}
    params = [
        "Events;2/MMcEvt_1./MMcEvt_1.fEvtNumber",
        "Events;2/MMcEvt_1./MMcEvt_1.fEnergy",
        "Events;2/MMcEvt_1./MMcEvt_1.fTheta",
        "Events;2/MMcEvt_1./MMcEvt_1.fPhi",
        "Events;2/MMcEvt_1./MMcEvt_1.fTelescopeTheta",
        "Events;2/MMcEvt_1./MMcEvt_1.fTelescopePhi",
        "Events;2/MMcEvt_1./MMcEvt_1.fZFirstInteraction",
        "Events;2/MMcEvt_2./MMcEvt_2.fEvtNumber",
        # Same as M1
        # "Events;2/MMcEvt_2./MMcEvt_2.fEnergy",
        # "Events;2/MMcEvt_2./MMcEvt_2.fTheta",
        # "Events;2/MMcEvt_2./MMcEvt_2.fPhi",
        # "Events;2/MMcEvt_2./MMcEvt_2.fTelescopeTheta",
        # "Events;2/MMcEvt_2./MMcEvt_2.fTelescopePhi",
        # "Events;2/MMcEvt_2./MMcEvt_2.fZFirstInteraction",
        "Events;2/MHillas_1./MHillas_1.fLength",
        "Events;2/MHillas_1./MHillas_1.fWidth",
        "Events;2/MHillas_1./MHillas_1.fDelta",
        "Events;2/MHillas_1./MHillas_1.fSize",
        "Events;2/MHillas_1./MHillas_1.fMeanX",
        "Events;2/MHillas_1./MHillas_1.fMeanY",
        "Events;2/MHillas_1./MHillas_1.fSinDelta",
        "Events;2/MHillas_1./MHillas_1.fCosDelta",
        "Events;2/MHillas_2./MHillas_2.fLength",
        "Events;2/MHillas_2./MHillas_2.fWidth",
        "Events;2/MHillas_2./MHillas_2.fDelta",
        "Events;2/MHillas_2./MHillas_2.fSize",
        "Events;2/MHillas_2./MHillas_2.fMeanX",
        "Events;2/MHillas_2./MHillas_2.fMeanY",
        "Events;2/MHillas_2./MHillas_2.fSinDelta",
        "Events;2/MHillas_2./MHillas_2.fCosDelta",
        "Events;2/MStereoPar./MStereoPar.fDirectionX",
        "Events;2/MStereoPar./MStereoPar.fDirectionY",
        "Events;2/MStereoPar./MStereoPar.fDirectionZd",
        "Events;2/MStereoPar./MStereoPar.fDirectionAz",
        "Events;2/MStereoPar./MStereoPar.fDirectionDec",
        "Events;2/MStereoPar./MStereoPar.fDirectionRA",
        "Events;2/MStereoPar./MStereoPar.fTheta2",
        "Events;2/MStereoPar./MStereoPar.fCoreX",
        "Events;2/MStereoPar./MStereoPar.fCoreY",
        "Events;2/MStereoPar./MStereoPar.fM1Impact",
        "Events;2/MStereoPar./MStereoPar.fM2Impact",
        "Events;2/MStereoPar./MStereoPar.fM1ImpactAz",
        "Events;2/MStereoPar./MStereoPar.fM2ImpactAz",
        "Events;2/MStereoPar./MStereoPar.fMaxHeight",
        "Events;2/MStereoPar./MStereoPar.fXMax",
        "Events;2/MStereoPar./MStereoPar.fCherenkovRadius",
        "Events;2/MStereoPar./MStereoPar.fCherenkovDensity",
        # Seems to be empty
        # "Events;2/MStereoPar./MStereoPar.fEnergy",
        # "Events;2/MStereoPar./MStereoPar.fEnergyUncertainty",
        # "Events;2/MStereoPar./MStereoPar.fEnergyDiscrepancy",
        "Events;2/MStereoPar./MStereoPar.fPhiBaseLineM1",
        "Events;2/MStereoPar./MStereoPar.fPhiBaseLineM2",
        "Events;2/MStereoPar./MStereoPar.fImageAngle",
        # Seems to be empty
        # "Events;2/MStereoPar./MStereoPar.fDispRMS",
        # "Events;2/MStereoPar./MStereoPar.fDispDiff2",
        "Events;2/MStereoPar./MStereoPar.fCosBSangle",
        "Events;2/MPointingPos_1./MPointingPos_1.fZd",
        "Events;2/MPointingPos_1./MPointingPos_1.fAz",
        # Same as M1
        # "Events;2/MPointingPos_2./MPointingPos_2.fZd",
        # "Events;2/MPointingPos_2./MPointingPos_2.fAz",
        "Events;2/MHillasTimeFit_1./MHillasTimeFit_1.fP1Grad",
        "Events;2/MHillasTimeFit_2./MHillasTimeFit_2.fP1Grad",
        "Events;2/MMcEvt_1./MMcEvt_1.fImpact",
        "Events;2/MMcEvt_2./MMcEvt_2.fImpact",
        "Events;2/MHillasSrc_1./MHillasSrc_1.fAlpha",
        "Events;2/MHillasSrc_1./MHillasSrc_1.fDist",
        "Events;2/MHillasSrc_1./MHillasSrc_1.fCosDeltaAlpha",
        "Events;2/MHillasSrc_1./MHillasSrc_1.fDCA",
        "Events;2/MHillasSrc_1./MHillasSrc_1.fDCADelta",
        "Events;2/MHillasSrc_2./MHillasSrc_2.fAlpha",
        "Events;2/MHillasSrc_2./MHillasSrc_2.fDist",
        "Events;2/MHillasSrc_2./MHillasSrc_2.fCosDeltaAlpha",
        "Events;2/MHillasSrc_2./MHillasSrc_2.fDCA",
        "Events;2/MHillasSrc_2./MHillasSrc_2.fDCADelta",
    ]

    with uproot.open(superstar_filepath) as f:
        # get the event number
        res["event_number"] = f["Events;2/MMcEvt_1./MMcEvt_1.fEvtNumber"].array(
            library="np"
        )
        res["run_number"] = np.full_like(res["event_number"], run_number)

        for k in params:
            title = k.split("/")[-1]
            res[title] = f[k].array().to_numpy()

        res["image_m1"] = [
            i.tojson() for i in f["Events;2/UprootImageOrig_1"].array(library="np")
        ]
        res["image_m2"] = [
            i.tojson() for i in f["Events;2/UprootImageOrig_2"].array(library="np")
        ]
        res["clean_image_m1"] = [
            i.tojson() for i in f["Events;2/UprootImageOrigClean_1"].array(library="np")
        ]
        res["clean_image_m2"] = [
            i.tojson() for i in f["Events;2/UprootImageOrigClean_2"].array(library="np")
        ]

    # also extract images and timing information from the calibrated files
    calibrated_res = {
        1: {},
        2: {},
    }
    for file, telescope in zip([calibrated_m1_file, calibrated_m2_file], [1, 2]):
        with uproot.open(file) as f:
            event_numbers = f["Events;1/MMcEvt./MMcEvt.fEvtNumber"].array(library="np")
            calibrated_res[telescope][f"calibrated_event_number_M{telescope}"] = (
                event_numbers.tolist()
            )
            n_events = len(event_numbers)

            # don't need to extract the images again
            # calibrated_res[telescope][f"calibrated_images_M{telescope}"] = (
            #     f["Events;1/MCerPhotEvt./MCerPhotEvt.fPixels/MCerPhotEvt.fPixels.fPhot"]
            #     .array(library="np")
            #     .tolist()
            # )

            arrival_times = []
            timing_data = f["Events;1/MArrivalTime./MArrivalTime.fData"].array()
            for t in timing_data:
                d = t.to_list()
                arrival_times.append(d)

            calibrated_res[telescope][f"calibrated_timing_M{telescope}"] = (
                np.array(arrival_times).reshape(n_events, len(d)).tolist()
            )

            # reformat calibrated res into {event_number: {m1_image: value, m2_image: value, ...}}
            event_key = f"calibrated_event_number_M{telescope}"
            calibrated_res[telescope] = {
                event_number: {k: v[i] for k, v in calibrated_res[telescope].items()}
                for i, event_number in enumerate(calibrated_res[telescope][event_key])
            }

            # remove event numbers over max_event_number
            calibrated_res[telescope] = {
                k: v
                for k, v in calibrated_res[telescope].items()
                if k < max_event_number
            }

    # merge based on event_number
    calibrated_merged = pd.DataFrame.from_dict(calibrated_res[1], orient="index").merge(
        pd.DataFrame.from_dict(calibrated_res[2], orient="index"),
        left_index=True,
        right_index=True,
    )

    # merge with the superstar data
    superstar_df = pd.DataFrame.from_dict(res)
    merged = superstar_df.merge(
        calibrated_merged, left_on="event_number", right_index=True
    )

    drop = [
        "MMcEvt_2.fEvtNumber",
        "MMcEvt_1.fEvtNumber",
        "calibrated_event_number_M1",
        "calibrated_event_number_M2",
    ]

    merged = merged.drop(columns=drop)

    rename_dict = {
        # Event Info
        "event_number": "event_number",
        "run_number": "run_number",
        # Monte Carlo Truth (both)
        "MMcEvt_1.fEnergy": "true_energy",
        "MMcEvt_1.fTheta": "true_theta",
        "MMcEvt_1.fPhi": "true_phi",
        "MMcEvt_1.fTelescopeTheta": "true_telescope_theta",
        "MMcEvt_1.fTelescopePhi": "true_telescope_phi",
        "MMcEvt_1.fZFirstInteraction": "true_first_interaction_height",
        # Monte Carlo Truth (M2)
        "MMcEvt_1.fImpact": "true_impact_m1",
        # Monte Carlo Truth (M2)
        "MMcEvt_2.fImpact": "true_impact_m2",
        # Hillas Parameters (M1)
        "MHillas_1.fLength": "hillas_length_m1",
        "MHillas_1.fWidth": "hillas_width_m1",
        "MHillas_1.fDelta": "hillas_delta_m1",
        "MHillas_1.fSize": "hillas_size_m1",
        "MHillas_1.fMeanX": "hillas_cog_x_m1",
        "MHillas_1.fMeanY": "hillas_cog_y_m1",
        "MHillas_1.fSinDelta": "hillas_sin_delta_m1",
        "MHillas_1.fCosDelta": "hillas_cos_delta_m1",
        # Hillas Parameters (M2)
        "MHillas_2.fLength": "hillas_length_m2",
        "MHillas_2.fWidth": "hillas_width_m2",
        "MHillas_2.fDelta": "hillas_delta_m2",
        "MHillas_2.fSize": "hillas_size_m2",
        "MHillas_2.fMeanX": "hillas_cog_x_m2",
        "MHillas_2.fMeanY": "hillas_cog_y_m2",
        "MHillas_2.fSinDelta": "hillas_sin_delta_m2",
        "MHillas_2.fCosDelta": "hillas_cos_delta_m2",
        # Stereo Parameters
        "MStereoPar.fDirectionX": "stereo_direction_x",
        "MStereoPar.fDirectionY": "stereo_direction_y",
        "MStereoPar.fDirectionZd": "stereo_zenith",
        "MStereoPar.fDirectionAz": "stereo_azimuth",
        "MStereoPar.fDirectionDec": "stereo_dec",
        "MStereoPar.fDirectionRA": "stereo_ra",
        "MStereoPar.fTheta2": "stereo_theta2",
        "MStereoPar.fCoreX": "stereo_core_x",
        "MStereoPar.fCoreY": "stereo_core_y",
        "MStereoPar.fM1Impact": "stereo_impact_m1",
        "MStereoPar.fM2Impact": "stereo_impact_m2",
        "MStereoPar.fM1ImpactAz": "stereo_impact_azimuth_m1",
        "MStereoPar.fM2ImpactAz": "stereo_impact_azimuth_m2",
        "MStereoPar.fMaxHeight": "stereo_shower_max_height",
        "MStereoPar.fXMax": "stereo_xmax",
        "MStereoPar.fCherenkovRadius": "stereo_cherenkov_radius",
        "MStereoPar.fCherenkovDensity": "stereo_cherenkov_density",
        "MStereoPar.fPhiBaseLineM1": "stereo_baseline_phi_m1",
        "MStereoPar.fPhiBaseLineM2": "stereo_baseline_phi_m2",
        "MStereoPar.fImageAngle": "stereo_image_angle",
        "MStereoPar.fCosBSangle": "stereo_cos_between_shower",
        # Pointing
        "MPointingPos_1.fZd": "pointing_zenith",
        "MPointingPos_1.fAz": "pointing_azimuth",
        # Time Fit
        "MHillasTimeFit_1.fP1Grad": "time_gradient_m1",
        "MHillasTimeFit_2.fP1Grad": "time_gradient_m2",
        # Source Parameters (M1)
        "MHillasSrc_1.fAlpha": "source_alpha_m1",
        "MHillasSrc_1.fDist": "source_dist_m1",
        "MHillasSrc_1.fCosDeltaAlpha": "source_cos_delta_alpha_m1",
        "MHillasSrc_1.fDCA": "source_dca_m1",
        "MHillasSrc_1.fDCADelta": "source_dca_delta_m1",
        # Source Parameters (M2)
        "MHillasSrc_2.fAlpha": "source_alpha_m2",
        "MHillasSrc_2.fDist": "source_dist_m2",
        "MHillasSrc_2.fCosDeltaAlpha": "source_cos_delta_alpha_m2",
        "MHillasSrc_2.fDCA": "source_dca_m2",
        "MHillasSrc_2.fDCADelta": "source_dca_delta_m2",
        # Images and Timing
        "image_m1": "image_m1",
        "image_m2": "image_m2",
        "clean_image_m1": "clean_image_m1",
        "clean_image_m2": "clean_image_m2",
        "calibrated_timing_M1": "timing_m1",
        "calibrated_timing_M2": "timing_m2",
    }

    # Apply the renaming
    merged = merged.rename(columns=rename_dict)

    if to_parquet:
        if not parquet_file_prefix:
            parquet_file_prefix = f"SuperStar_{run_number}"

        output_filename = superstar_output_dir / f"{parquet_file_prefix}.parquet"
        merged.to_parquet(output_filename)
        return output_filename

    return merged


def process_row(
    row,
    output_dir: str | Path,
    run_number_col: str = "run_number",
    m1_filepath_col: str = "m1_calib_file",
    m2_filepath_col: str = "m2_calib_file",
):
    """Process a single row from a dataframe with self-contained environment setup"""
    # Import required modules inside function for worker processes

    # Create a unique output directory for this process
    # process_id = uuid.uuid4().hex[:8]
    # process_output_dir = output_dir / f"process_{process_id}"
    # process_output_dir.mkdir(parents=True, exist_ok=True)

    if "run_number" not in row.index:
        raise ValueError("Run number not found in row")

    # creat a unique output directory for this run number
    process_output_dir = output_dir / f"SuperStar-{row[run_number_col]}"
    process_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Run the processing
        result_file = process_calibrated(
            run_number=row[run_number_col],
            calibrated_m1_file=row[m1_filepath_col],
            calibrated_m2_file=row[m2_filepath_col],
            star_output_dir=process_output_dir,
            to_parquet=True,
        )

        # Move the result file to the main output directory
        final_path = output_dir / result_file.name
        shutil.move(str(result_file), str(final_path))

        # Clean up the temporary directory
        shutil.rmtree(process_output_dir)

        return final_path

    except Exception as e:
        # Clean up on error
        if process_output_dir.exists():
            shutil.rmtree(process_output_dir)
        raise e
