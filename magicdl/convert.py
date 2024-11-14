from pathlib import Path

import numpy as np
import pandas as pd
import uproot
import dotenv
import subprocess
import magicpy.mars as mars


def extract_calibrated_mc(
    filepath: Path | str, verbose: bool = False
) -> tuple[dict, dict]:
    """Extract event data from a calibrated MAGIC ROOT MC file.

    Args:
        filepath (Path | str): Input file path to a calibrated MAGIC ROOT file.
        verbose (bool, optional): Whether or not to print summary info for each file. Defaults to False.

    Returns:
        tuple[dict, dict]: Returns a tuple of two dictionaries. The first dictionary contains overall run parameters, and the second dictionary contains event parameters and images.
    """

    filepath = Path(filepath).absolute()

    # interesting run headers

    # have a length of 1000?  so even non-triggered events are stored?
    original = [
        # "OriginalMC;1/MMcEvtBasic./MMcEvtBasic.fEnergy",
        # "OriginalMC;1/MMcEvtBasic./MMcEvtBasic.fImpact",
        # "OriginalMC;1/MMcEvtBasic./MMcEvtBasic.fTelescopePhi",
        # "OriginalMC;1/MMcEvtBasic./MMcEvtBasic.fTelescopeTheta",
    ]

    headers = [
        "RunHeaders;1/MRawRunHeader./MRawRunHeader.fTelescopeNumber",
        # "RunHeaders;1/MMcRunHeader./MMcRunHeader.fNumEvents",  # this is num simulated, not num triggered
        "RunHeaders;1/MRawRunHeader./MRawRunHeader.fRunNumber",
    ]

    # events
    event_info = [
        "Events;1/MMcEvt./MMcEvt.fEvtNumber",
        "Events;1/MMcEvt./MMcEvt.fEnergy",
        "Events;1/MMcEvt./MMcEvt.fZFirstInteraction",
        "Events;1/MMcEvt./MMcEvt.fTheta",
        "Events;1/MMcEvt./MMcEvt.fPhi",
        "Events;1/MMcEvt./MMcEvt.fTelescopeTheta",
        "Events;1/MMcEvt./MMcEvt.fTelescopePhi",
        # 'Events;1/MMcEvt./MMcEvt.fCoreD',
        # 'Events;1/MMcEvt./MMcEvt.fCoreX',
        # 'Events;1/MMcEvt./MMcEvt.fCoreY',
        'Events;1/MMcEvt./MMcEvt.fImpact',
        # 'Events;1/MMcEvt./MMcEvt.fTimeFirst',
        # 'Events;1/MMcEvt./MMcEvt.fTimeLast',
        # p.e. image data
        "Events;1/MCerPhotEvt./MCerPhotEvt.fPixels/MCerPhotEvt.fPixels.fPhot",
        # what are these two?
        # 'Events;1/MPedPhotFundamental./MPedPhotFundamental.fArray/MPedPhotFundamental.fArray.fMean',
        # 'Events;1/MPedPhotFundamental./MPedPhotFundamental.fArray/MPedPhotFundamental.fArray.fRms',
    ]

    with uproot.open(filepath) as f:
        overall_params = {}

        for k in original:
            title = k.split(".")[-1]
            overall_params[title] = f[k].array()[0]

        for k in headers:
            title = k.split(".")[-1]
            overall_params[title] = f[k].array()[0]

        event_params = {}

        for k in event_info:
            title = k.split(".")[-1]
            
            try:
                event_params[title] = f[k].array().to_numpy()
            except ValueError as e:
                # some file can be buggy have irregular lengths
                if verbose:
                    print(f"Error extracting {title} from {filepath}: {e}")
                overall_params['has_error'] = True
                overall_params['fNumTrig'] = 0
                return overall_params, {}

    n_events = len(event_params["fEvtNumber"])
    overall_params["fNumTrig"] = n_events

    arrival_times = []
    timing_data = f["Events;1/MArrivalTime./MArrivalTime.fData"].array()
    for t in timing_data:
        d = t.to_list()
        arrival_times.append(d)

    event_params["MArrivalTime"] = np.array(arrival_times).reshape(n_events, len(d))

    # generate event ID from run number and event number
    event_ids = []

    for i in range(n_events):
        event_ids.append(
            f"{overall_params['fRunNumber']}.{event_params['fEvtNumber'][i]}"
        )

    event_params["event_id"] = np.array(event_ids)

    if verbose:
        print(
            f"Telescope {overall_params['fTelescopeNumber']} / Run {overall_params['fRunNumber']}"
        )
        print(f"Extracted {n_events} events from {filepath}")
        print("--- Overall params: ---")
        for k, v in overall_params.items():
            print(k, v)

        print("\n--- Event params: ---")
        for k, v in event_params.items():
            print(k, v.shape)
        print()

    # convert ndarrays to lists
    for k, v in event_params.items():
        if isinstance(v, np.ndarray):
            event_params[k] = v.tolist()

    return overall_params, event_params


def merge_calibrated_mc(
    m1_filepath: Path | str,
    m2_filepath: Path | str,
    stereo_only: bool = True,
    export: bool = True,
    export_filepath: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame | Path | None:
    """Extract and merge two calibrated MAGIC ROOT MC files into a single DataFrame.

    Args:
        m1_filepath (Path | str): Path to the M1 MAGIC ROOT MC file.
        m2_filepath (Path | str): Path to the M2 MAGIC ROOT MC file.
        stereo_only (bool, optional): Whether to filter out non-stereo stereo events. Defaults to True.
        export (bool, optional): Whether to export the merged DataFrame to a parquet file. If False, a DataFrame is returned. Defaults to False.
        export_filepath (Path | None, optional): Path to export the merged DataFrame. If not provided, a file is saved in the current directory. Defaults to None.
        verbose (bool, optional): Whether to print summary info for each file. Defaults to False.

    Returns:
        pd.DataFrame | Path: _description_
    """
    m1_filepath = Path(m1_filepath).absolute()
    m2_filepath = Path(m2_filepath).absolute()

    o1, e1 = extract_calibrated_mc(m1_filepath, verbose=verbose)
    o2, e2 = extract_calibrated_mc(m2_filepath, verbose=verbose)
    
    if o1.get('has_error') or o2.get('has_error'):
        if verbose and o1.get('has_error'):
            print(f"Error extracting data from {m1_filepath}")
        if verbose and o2.get('has_error'):
            print(f"Error extracting data from {m2_filepath}")
        return

    # turn each into a dataframe
    df1 = pd.DataFrame(e1)

    # add fTelecopeNumber and fRunNumber to each event
    # df1['fTelescopeNumber'] = o1['fTelescopeNumber']
    df1["fRunNumber"] = o1["fRunNumber"]

    df2 = pd.DataFrame(e2)

    # add fTelecopeNumber and fRunNumber to each event
    # df2['fTelescopeNumber'] = o2['fTelescopeNumber']
    df2["fRunNumber"] = o2["fRunNumber"]

    # drop all rows with event number == 1000
    # for some reason 1000 is always glitchy
    df1 = df1[df1.fEvtNumber != 1000]
    df2 = df2[df2.fEvtNumber != 1000]

    merge = pd.merge(
        df1,
        df2,
        on=[
            "event_id",
            "fEvtNumber",
            "fRunNumber",
            "fEnergy",
            "fTheta",
            "fPhi",
            "fTelescopeTheta",
            "fTelescopePhi",
            "fZFirstInteraction",
        ],
        suffixes=("M1", "M2"),
        how="inner" if stereo_only else "outer",
    )

    # order the columns
    all_cols = list(merge.columns)
    first_cols = [
        "event_id",
        "fRunNumber",
        "fEvtNumber",
        "fEnergy",
        "fTheta",
        "fPhi",
        "fTelescopeTheta",
        "fTelescopePhi",
        "fZFirstInteraction",
    ]

    merge = merge[first_cols + [c for c in all_cols if c not in first_cols]]

    if export and export_filepath is None:
        run_number = o1["fRunNumber"]
        file_stem = f"MC_run_{run_number}"

        if stereo_only:
            file_stem += "_stereo"
        export_filepath = Path(f"./{file_stem}.parquet").absolute()

    if export:
        merge.to_parquet(export_filepath)
        return export_filepath
    else:
        return merge


def process_calibrated(
    run_number: int,
    calibrated_m1_file: Path | str,
    calibrated_m2_file: Path | str,
    star_output_dir: Path | str,
    superstar_output_dir: Path | str | None = None,
    to_parquet: bool = True,
):

    # load in environment variables
    env = dict(dotenv.dotenv_values())
    
    if not env.get("MARSSYS"):
        raise ValueError("MARSSYS environment variable not set")
    if not env.get("ROOTSYS"):
        raise ValueError("ROOTSYS environment variable not set")
    
    if not superstar_output_dir:
        superstar_output_dir = star_output_dir
    
    if not isinstance(calibrated_m1_file, Path):
        calibrated_m1_file = Path(calibrated_m1_file)
    if not isinstance(calibrated_m2_file, Path):
        calibrated_m2_file = Path(calibrated_m2_file)
    
    starm1 = mars.Star(
        input_files=[calibrated_m1_file],
        analysis_path=star_output_dir,
        output_dir=star_output_dir,
        telescope_number=1,
        is_mc=True,
    )

    starm2 = mars.Star(
        input_files=[calibrated_m2_file],
        analysis_path=star_output_dir,
        output_dir=star_output_dir,
        telescope_number=2,
        is_mc=True,
    )
    
    resm1 = starm1.run(
        file=calibrated_m1_file,
        save_all=True,
        overwrite=True,
    )

    resm2 = starm2.run(
        file=calibrated_m2_file,
        save_all=True,
        overwrite=True,
    )
    
    star_m1_filepath = resm1["star_file_path"]
    star_m2_filepath = resm2["star_file_path"]
    
    star_m1_regex = star_m1_filepath.parent / f"{star_m1_filepath.stem}*.root"
    star_m2_regex = star_m2_filepath.parent / f"{star_m2_filepath.stem}*.root"

    superstar = mars.SuperStar(
        input_files=[star_m1_regex, star_m2_regex],
        analysis_path=superstar_output_dir,
        output_dir=superstar_output_dir,
        is_mc=True,
    )
    
    ssres = superstar.run(
        m1_file=star_m1_regex,
        m2_file=star_m2_regex,
        is_mc=True,
        overwrite=True,
    )
    
    superstar_filepath = ssres["superstar_file"]
    
    # run the writeImages2uproot macro
    eof = f'root -b -l -q {env["MARSSYS"]}/macros/writeImages2uproot.C("{superstar_filepath.name}")'
    
    process = subprocess.Popen(
        eof.split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=superstar_output_dir,
        env=ssres.env | env,
    )
    
    output, error = process.communicate()
    
    if process.returncode != 0:
        raise ValueError(f"Error running writeImages2uproot.C: {error}")
    
    res = {}
    
    params = [
        'Events;2/MMcEvt_1./MMcEvt_1.fEvtNumber',
        'Events;2/MMcEvt_1./MMcEvt_1.fEnergy',
        'Events;2/MMcEvt_1./MMcEvt_1.fTheta',
        'Events;2/MMcEvt_1./MMcEvt_1.fPhi',
        'Events;2/MMcEvt_1./MMcEvt_1.fTelescopeTheta',
        'Events;2/MMcEvt_1./MMcEvt_1.fTelescopePhi',
        'Events;2/MMcEvt_1./MMcEvt_1.fZFirstInteraction',
        'Events;2/MMcEvt_2./MMcEvt_2.fEvtNumber',
        'Events;2/MMcEvt_2./MMcEvt_2.fEnergy',
        'Events;2/MMcEvt_2./MMcEvt_2.fTheta',
        'Events;2/MMcEvt_2./MMcEvt_2.fPhi',
        'Events;2/MMcEvt_2./MMcEvt_2.fTelescopeTheta',
        'Events;2/MMcEvt_2./MMcEvt_2.fTelescopePhi',
        'Events;2/MMcEvt_2./MMcEvt_2.fZFirstInteraction',
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
        # "Events;2/MStereoPar./MStereoPar.fEnergy",
        # "Events;2/MStereoPar./MStereoPar.fEnergyUncertainty",
        # "Events;2/MStereoPar./MStereoPar.fEnergyDiscrepancy",
        "Events;2/MStereoPar./MStereoPar.fPhiBaseLineM1",
        "Events;2/MStereoPar./MStereoPar.fPhiBaseLineM2",
        "Events;2/MStereoPar./MStereoPar.fImageAngle",
        # "Events;2/MStereoPar./MStereoPar.fDispRMS",
        # "Events;2/MStereoPar./MStereoPar.fDispDiff2",
        "Events;2/MStereoPar./MStereoPar.fCosBSangle",
        "Events;2/MPointingPos_2./MPointingPos_2.fZd",
        "Events;2/MPointingPos_2./MPointingPos_2.fAz",
        'Events;2/MHillasTimeFit_1./MHillasTimeFit_1.fP1Grad',
        'Events;2/MHillasTimeFit_2./MHillasTimeFit_2.fP1Grad',
        'Events;2/MMcEvt_1./MMcEvt_1.fImpact',
        'Events;2/MMcEvt_2./MMcEvt_2.fImpact',
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
    
    res["event_number"] = f["Events;2/MMcEvt_1./MMcEvt_1.fEvtNumber"].array(library="np")
    res["run_number"] = np.full_like(res["event_number"], run_number)
    
    
    with uproot.open(superstar_filepath) as f:
        
        for k in params:
            title = k.split('/')[-1]
            res[title] = f[k].array().to_numpy()
        
        res["images_m1"] = [i.tojson() for i in f['Events;2/UprootImageOrig_2'].array(library='np')]
        res["images_m2"] = [i.tojson() for i in f['Events;2/UprootImageOrigClean_1'].array(library='np')]
        res["clean_images_m1"] = [i.tojson() for i in f['Events;2/UprootImageOrigClean_2'].array(library='np')]
        res["clean_images_m2"] = f["Events;2/MMcEvt_1./MMcEvt_1.fEvtNumber"].array(library="np")
        
    if to_parquet:
        df = pd.DataFrame.from_dict(res)
        output_filename = superstar_output_dir / f"{superstar_filepath.stem}.parquet"
        df.to_parquet(output_filename)
        return output_filename
    
    return res
