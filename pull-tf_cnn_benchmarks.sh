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
#
# official/utils/logs
#   versions (branches):
#   r1.10.0, r1.11, r1.12.0, r1.13.0, r2.1.0, r2.2.0, r2.3.0, r2.4.0
#
# tf_cnn_benchmarks 
#   versions (branches): 
#   cnn_tf_v{1.10, 1.11, 1.12, 1.13, 1.14, 1.15, 2.0, 2.1}_compatible
# 
# Now we use/support versions TF1.14, TF1.15, TF2.0, TF2.1
###

##### USAGEMESSAGE #####
USAGEMESSAGE="Usage: sh $0 --tf_ver --tfbench_path \n
	    --tf_ver=\"2.1\" - is a corresponding TF Version, e.g. 1.14, 2.1 etc \n
        --tfbench_path=\"../tf_cnn_benchmarks\" - is where to install TF Benchmarks. \n
        By default TF Benchmarks are installed above directory the script is \n
        called from, i.e. \"../tf_cnn_benchmarks\""

TFVer="1.14"
TFBenchPATH="../tf_cnn_benchmarks"

##### TRY TO DEDUCE TensorFlow version #####
# !! python3 and tensorflow have to be installed !!
if [ -e /usr/bin/python3 ]; then
    TFVer=$(python3 -c "import tensorflow; print(tensorflow.__version__)")
    TFVer=$(echo ${TFVer} | cut -d\. -f1,2)
fi

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
        [[ $i = *"--tfbench_path"* ]]  && TFBenchPATH=${i#*=}
    done
else
    # Too many arguments were given (>3)
    echo "ERROR! Too many arguments provided!"
    shopt -s xpg_echo    
    echo $USAGEMESSAGE
    exit 2
fi

echo "[INFO] Configured TensorFlow version: $TFVer"
TFVerMain=$(echo ${TFVer} | cut -d\. -f1)

### Script full path
# https://unix.stackexchange.com/questions/17499/get-path-of-current-script-when-executed-through-a-symlink/17500
SCRIPT_PATH="$(dirname "$(readlink -f "$0")")"

### Check if TMP directory exists
TMP_PATH="${SCRIPT_PATH}/tmp"
if [ ! -d "$TMP_PATH" ]; then
   mkdir $TMP_PATH
fi

### TF BENCHMARKS ###
#Clone TF Benchmarks
TFBenchTMP="benchmarks.tmp"  # temporary dir to clone git repository

TFVerTF=${TFVer}
# !!! FOR 2.4.0, 2.3.0, 2.2.0 THERE IS NO CORRESPONDING BRANCH, USE 2.1.0 !!!
if [ "$TFVer" = 2.4 ] || [ "$TFVer" = 2.3 ] || [ "$TFVer" = 2.2 ]; then
    TFVerTF="2.1"
    TFVerOff="2.1"
fi
echo "[INFO] Installing TF Benchmarks for TF${TFVerTF} in ${TFBenchPATH}"

# Switch to TMP directory, check if we can clone benchmarks there
cd ${TMP_PATH}
if [ -d "${TFBenchTMP}" ]; then
   rm -rf ${TFBenchTMP}
fi
git clone --depth 1 -b cnn_tf_v${TFVerTF}_compatible https://github.com/tensorflow/benchmarks.git ${TFBenchTMP}

cd ${SCRIPT_PATH} && mv -T ${TMP_PATH}/${TFBenchTMP}/scripts/tf_cnn_benchmarks ${TFBenchPATH} && \
rm -rf ${TMP_PATH}/${TFBenchTMP}

# If necessary, apply patch on tf_cnn_benchmarks (normal way)
TF_CNN_PATCH="${SCRIPT_PATH}/patches/tf_cnn_benchmarks_${TFVerTF}.patch"
if test -f ${TF_CNN_PATCH}; then
    cd ${TFBenchPATH} &&
    echo "[INFO] Applying ${TF_CNN_PATCH} in ${TFBenchPATH}" && \
    patch < ${TF_CNN_PATCH}
else
    TF_CNN_PATCH="${SCRIPT_PATH}/patches/tf_cnn_benchmarks_${TFVerMain}.patch"
    if test -f ${TF_CNN_PATCH}; then
        cd ${TFBenchPATH} &&
        echo "[INFO] Applying ${TF_CNN_PATCH} in ${TFBenchPATH}" && \
        patch < ${TF_CNN_PATCH}
    fi
fi

### Official/utils/logs ###
# Clone models/official/utils/logs
TFModelsTMP="models.tmp"  # temporary dir to clone git repository

TFVerOff=${TFVer}
# !!! FOR 1.14 and 1.15 THERE IS NO CORRESPONDING BRANCH, USE r1.13.0 !!!
if [ "$TFVer" = 1.14 ] || [ "$TFVer" = 1.15 ]; then
        TFVerOff="1.13"
fi
# !!! FOR 2.0 THERE IS NO CORRESPONDING BRANCH, USE r2.1.0 !!!
if [ "$TFVer" = 2.0 ]; then
        TFVerOff="2.1"
fi

echo "[INFO] Installing models/official/utils/logs for TF${TFVerOff} under ${TFBenchPATH}"

# Swtich to TMP directory, check if we can clone models/official there
cd ${TMP_PATH}
if [ -d "${TFModelsTMP}" ]; then
   rm -rf ${TFModelsTMP}
fi
mkdir ${TFModelsTMP} && cd ${TFModelsTMP} && git init && \
git remote add origin https://github.com/tensorflow/models.git && \
git fetch --depth 1 origin && \
git checkout origin/r${TFVerOff}.0 official/utils/logs

# If necessary, apply patch
OFFICIAL_LOGGER_PATCH="${SCRIPT_PATH}/patches/official_logger_${TFVerOff}.patch"
if test -f ${OFFICIAL_LOGGER_PATCH}; then
    echo "[INFO] Applying ${OFFICIAL_LOGGER_PATCH} patch" && \
    git apply ${OFFICIAL_LOGGER_PATCH}
else
    OFFICIAL_LOGGER_PATCH="${SCRIPT_PATH}/patches/official_logger_${TFVerMain}.patch"
    if test -f ${OFFICIAL_LOGGER_PATCH}; then
        echo "[INFO] Applying ${OFFICIAL_LOGGER_PATCH} patch" && \
        git apply ${OFFICIAL_LOGGER_PATCH}
    fi
fi

cd ${SCRIPT_PATH} && mv ${TMP_PATH}/${TFModelsTMP}/official ${TFBenchPATH}
rm -rf ${TMP_PATH}/${TFModelsTMP}
###

### Delete TMP_PATH, if empty
[[ "$(ls -A ${TMP_PATH})" ]] && echo "[WARNING] ${TMP_PATH} is NOT empty!" || rm -rf ${TMP_PATH}

### Info message
TFBenchFullPATH=$(cd ${TFBenchPATH} && pwd)
echo ""
echo "===========>>>"
echo " You may need to add ${TFBenchFullPATH} to PYTHONPATH environment! "
echo "===========>>>"
