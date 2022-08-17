#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
find $SCRIPT_DIR/include/ $SCRIPT_DIR/src/ -iname *.hpp -o -iname *.cpp | xargs clang-format -i