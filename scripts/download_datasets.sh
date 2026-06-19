#!/usr/bin/env bash
# Download the free, small (<10 GB total) public datasets used by the
# CAREER_ROADMAP. All are resumable (curl -C -); safe to re-run.
#
#   bash scripts/download_datasets.sh            # download everything below
#   bash scripts/download_datasets.sh tumvi      # only the TUM-VI group
#   bash scripts/download_datasets.sh euroc       # only EuRoC
#   bash scripts/download_datasets.sh tumrgbd     # only TUM RGB-D
#
# Files land under ./datasets/<group>/ (git-ignored). Archives are extracted
# in place and the tarball kept (delete by hand to reclaim space).
set -uo pipefail
cd "$(dirname "$0")/.."
DEST="datasets"

get () {  # get <url> <out_relpath>
  local url="$1" out="$DEST/$2"
  mkdir -p "$(dirname "$out")"
  if [ -f "$out" ]; then echo "  ✓ have $out"; fi
  echo "  ↓ $url"
  curl -L --fail --retry 3 --retry-delay 5 -C - -o "$out" "$url" \
    || { echo "  ✗ FAILED: $url"; return 1; }
}

extract () {  # extract <archive_relpath>
  local a="$DEST/$1"
  case "$a" in
    *.tar)      tar -xf "$a"   -C "$(dirname "$a")" ;;
    *.tgz|*.tar.gz) tar -xzf "$a" -C "$(dirname "$a")" ;;
    *.zip)      unzip -n -q "$a" -d "${a%.zip}" ;;
  esac
}

dl_tumvi () {
  echo "== TUM-VI (fisheye stereo + IMU + mocap GT + published Double Sphere calib) =="
  local base="https://cdn3.vision.in.tum.de/tumvi/exported/euroc/512_16"
  get "$base/dataset-room1_512_16.tar"      tumvi/dataset-room1_512_16.tar      && extract tumvi/dataset-room1_512_16.tar
  get "$base/dataset-calib-cam1_512_16.tar" tumvi/dataset-calib-cam1_512_16.tar && extract tumvi/dataset-calib-cam1_512_16.tar
  get "$base/dataset-calib-imu1_512_16.tar" tumvi/dataset-calib-imu1_512_16.tar && extract tumvi/dataset-calib-imu1_512_16.tar
}

dl_tumrgbd () {
  echo "== TUM RGB-D (RGB+depth+GT pose; for learned-depth metric validation) =="
  get "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz" \
      tumrgbd/rgbd_dataset_freiburg1_xyz.tgz && extract tumrgbd/rgbd_dataset_freiburg1_xyz.tgz
}

dl_euroc () {
  echo "== EuRoC MAV — Vicon Room 1 bundle (stereo + IMU + GT + radtan calib) =="
  # EuRoC migrated to the ETH Research Collection, which serves bundled ZIPs via a
  # browser/JS download (no scriptable URL). Supply the "Vicon Room 1 Datasets"
  # direct link once, then:
  #     EUROC_VR1='<pasted-url>' bash scripts/download_datasets.sh euroc
  : "${EUROC_VR1:=}"
  if [ -n "$EUROC_VR1" ]; then
    get "$EUROC_VR1" euroc/vicon_room1.zip && extract euroc/vicon_room1.zip
  else
    echo "  (EUROC_VR1 not set — right-click 'Vicon Room 1 Datasets' on the ETH"
    echo "   Research Collection page, Copy link, and re-run with EUROC_VR1=<url>.)"
  fi
}

case "${1:-all}" in
  tumvi)   dl_tumvi ;;
  tumrgbd) dl_tumrgbd ;;
  euroc)   dl_euroc ;;
  all)     dl_tumvi; dl_tumrgbd; dl_euroc ;;
  *) echo "usage: $0 [all|tumvi|tumrgbd|euroc]"; exit 1 ;;
esac
echo "Done. Total size:"; du -sh "$DEST" 2>/dev/null
