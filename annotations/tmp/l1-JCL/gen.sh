# item SEG PERSONS STAT TURN "CAPTION" "ref1 ref2 ..."
item() {
  local refs=""; for r in ${=6}; do refs="$refs\"JCL/$r\","; done; refs="[${refs%,}]"
  printf '{"segment_id":"JCL:%s","persons_in_corridor":"%s","stationary_group":"TRUTH_%s","doorway_traversal":"TRUTH_FALSE","vehicle_present":"TRUTH_FALSE","vehicle_interaction":"TRUTH_FALSE","bicycle":"TRUTH_FALSE","indoor":"TRUTH_TRUE","turn_visible":"TRUTH_%s","caption":"%s","refs":%s}' "$1" "$2" "$3" "$4" "$5" "$refs"
}
