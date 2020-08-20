#!/usr/bin/env bash
#
# Copyright (c) 2017 - 2019 Karlsruhe Institute of Technology - Steinbuch Centre for Computing
# This code is distributed under its License. Please, see the LICENSE file
#
# @author: vykozlov
#
###
# Script to pull TF Benchmarks and necessary offical/utils/logs scripts
# Applies a few patches depending on TF version.
###

##### USAGEMESSAGE #####
USAGEMESSAGE="Usage: sh $0 --tf_ver --tfbench_path \n
	    --tf_ver=\"2.1\" - is a corresponding TF Version, e.g. 1.14, 2.1 etc \n
        --tfbench_path=\"../tf_cnn_benchmarks\" - is where to install TF Benchmarks. \n
        By default TF Benchmarks are installed above directory the script is \n
        called from, i.e. \"../tf_cnn_benchmarks\""

TFVer="1.14"
TFBenchPATH="../tf_cnn_benchmarks"

##### PARSE SCRIPT FLAGS #####
arr=("$@")
if [ $# -eq 0 ]; then 
# use default config (0)
    break 
elif [ $1 == "-h" ] || [ $1 == "--help" ]; then 
# print usagemessage
    shopt -s xpg_echo
    echo $USAGEMESSAGE
    exit 1
elif [ $# -ge 1 ] && [ $# -le 2 ]; then 
# read benchmark options as parameters (1-2)
    for i in "${arr[@]}"; do
        [[ $i = *"--tf_ver"* ]]  && TFVer=${i#*=} 
        [[ $i = *"--tfbench_dir"* ]]  && TFBenchPATH=${i#*=}
    done
else
    # Too many arguments were given (>3)
    echo "ERROR! Too many arguments provided!"
    shopt -s xpg_echo    
    echo $USAGEMESSAGE
    exit 2
fi

# Script full path
# https://unix.stackexchange.com/questions/17499/get-path-of-current-script-when-executed-through-a-symlink/17500
SCRIPT_PATH="$(dirname "$(readlink -f "$0")")"
TMP_PATH="${SCRIPT_PATH}/tmp"

# Check if TMP directory exists
if [ ! -d "$TMP_PATH" ]; then
   mkdir $TMP_PATH
fi

# Clone TF Benchmarks
echo "[INFO] Installing TF Benchmarks in ${TFBenchPATH}"
TFBenchTMP="benchmarks.tmp"  # temporary dir to clone git repository

if [ "$TFVer" = 2.3 ] || [ "$TFVer" = 2.2 ]; then
    TFVer="2.1"
fi

# Swtich to TMP directory, check if we can clone benchmarks there
cd ${TMP_PATH}
if [ -d "${TFBenchTMP}" ]; then
   rm -rf ${TFBenchTMP}
fi
git clone --depth 1 -b cnn_tf_v${TFVer}_compatible https://github.com/tensorflow/benchmarks.git ${TFBenchTMP}

cd ${SCRIPT_PATH} && mv -T ${TMP_PATH}/${TFBenchTMP}/scripts/tf_cnn_benchmarks ${TFBenchPATH} && \
rm -rf ${TMP_PATH}/${TFBenchTMP}

# If necessary, apply patch on tf_cnn_benchmarks (normal way)
TF_CNN_PATCH="${SCRIPT_PATH}/patches/tf_cnn_benchmarks_${TFVer}.patch"
if test -f ${TF_CNN_PATCH}; then
    cd ${TFBenchPATH} &&
    echo "[INFO] Applying ${TF_CNN_PATCH} in ${TFBenchPATH}" && \
    patch < ${TF_CNN_PATCH}
fi

# Clone models/official/utils/logs
echo "[INFO] Installing models/official/utils/logs under ${TFBenchPATH}"
TFModelsTMP="models.tmp"  # temporary dir to clone git repository

# !!! FOR 1.14 and 1.15 THERE IS NO CORRESPONDING BRANCH, USE r1.13.0 !!!
if [ "$TFVer" = 1.14 ] || [ "$TFVer" = 1.15 ]; then
        TFVer="1.13"
fi

# Swtich to TMP directory, check if we can clone models/official there
cd ${TMP_PATH}
if [ -d "${TFModelsTMP}" ]; then
   rm -rf ${TFModelsTMP}
fi
mkdir ${TFModelsTMP} && cd ${TFModelsTMP} && git init && \
git remote add origin https://github.com/tensorflow/models.git && \
git fetch --depth 1 origin && \
git checkout origin/r${TFVer}.0 official/utils/logs

# If necessary, apply patch
OFFICIAL_LOGGER_PATCH="${SCRIPT_PATH}/patches/official_logger_${TFVer}.patch"
if test -f ${OFFICIAL_LOGGER_PATCH}; then
    echo "[INFO] Applying ${OFFICIAL_LOGGER_PATCH} patch" && \
    git apply ${OFFICIAL_LOGGER_PATCH}
fi

cd ${SCRIPT_PATH} && mv ${TMP_PATH}/${TFModelsTMP}/official ${TFBenchPATH}
rm -rf ${TMP_PATH}/${TFModelsTMP}

# Delete TMP_PATH, if empty
[[ "$(ls -A ${TMP_PATH})" ]] && echo "[WARNING] ${TMP_PATH} is NOT empty!" || rm -rf ${TMP_PATH}

