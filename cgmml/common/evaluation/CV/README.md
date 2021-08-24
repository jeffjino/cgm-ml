# Computer vision

## Setting up the environment
To set up the environment, you have to be a member of CGM development team and have permissions to the Azure Storage and Cloudbeaver.

### Exporting metadata
Open Cloudbeaver <https://cgm-core-infra-prod-ci-sqlmgmt.azurewebsites.net/#/> and run following SQL query:

`select s.id as scan_id, a.ord as ord, a.id as artifact_id, f.file_path as file_path, f.file_extension as file_extension, m.height, m.weight, m.muac,  s.version as scan_version, s.scan_type_id as scan_type, m.measured as measurement_date, s.scan_start as scan_date from scan s
join measure m on s.person_id = m.person_id and s.scan_start::date = m.measured::date
join artifact a on a.scan_id = s.id
join file f on a.file_id = f.id`

Then click on export and export as csv.

### Downloading raw data
You can browse the data using Microsoft Azure Storage Explorer under `cgm-in-bmz-sub/cgmbeciinbmzsa/Blob Containers/cgm-raw`.

Note that setting up the download process takes about 2hours and downloading takes more than 60h.

To download more than one page of data in Microsoft Azure Storage Explorer:
* click on load more so often until it loads the whole selection
* click on select all/select all cached
* click on download

Known limitations:
* downloading 75000 files/folders at once immediatelly crashes the window tab
* downloading 20000 files/folders at once uses more than 16GB RAM and it is much slower than downloading smaller pieces (due to RAM swapping)
You can use the filter function to limit the downloaded amounts (the current size of the raw data is 17.5GB)

### Generate RGBD matching for metadata
Run
`python rgbd_match.py metadata_path`
where metadata_path is the path to the exported metadata file

Copy the generated `newmetadata.csv` file into the raw data folder.

## Using the evaluation system

### Running the evaluation
Run
`python evaluation.py rawdata_path newmetadata.csv`
where rawdata_path is the path to the download raw data

This will take about 30-45 minutes (depending on data size and device performance). The results will be generated stored in these files:
* output.csv - estimation result and error for every item
* rejections.csv - rejected scans with reasons for rejection
* report.csv - report of the evaluation statistics

### Visualising the data
To visualize the processed data run
`python renderer.py rawdata_path output.csv`
where rawdata_path is the path to the download raw data

To visualize the rejected data run
`python renderer.py rawdata_path rejections.csv`
where rawdata_path is the path to the download raw data

In both cases the visualisations will be stored in data/render of root of the repository
