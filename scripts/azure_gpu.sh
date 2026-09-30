#!/usr/bin/env bash
# BREE Azure GPU VMs. Everything in resource group bree-rg, tagged project=bree, auto-shutdown on,
# SSH only and only from this machine's current public IP.
#   scripts/azure_gpu.sh quota  [region]   # GPU quota in a region
#   scripts/azure_gpu.sh train  [region]   # bree-train: 1x A100 80GB (NC24ads_A100_v4), 1 TB disk, driver, repo, make test
#   scripts/azure_gpu.sh sim    [region]   # bree-sim:   1x A10 24GB  (NV36ads_A10_v5) via NVIDIA Isaac Automator
#   scripts/azure_gpu.sh stop              # deallocate every VM in bree-rg (keeps disks, stops compute billing)
#   scripts/azure_gpu.sh status
set -euo pipefail
RG=bree-rg; TAGS="project=bree"; REGION="${2:-eastus}"
SHUTDOWN_UTC="${SHUTDOWN_UTC:-0900}"        # daily auto-shutdown time (UTC hhmm)
MYIP="$(curl -s https://api.ipify.org)"

lock_ssh() {   # $1 = vm name: replace the default SSH rule with one locked to this machine
  local nsg; nsg=$(az network nsg list -g $RG --query "[?contains(name,'$1')].name | [0]" -o tsv)
  az network nsg rule list -g $RG --nsg-name "$nsg" --query "[].name" -o tsv | xargs -r -n1 az network nsg rule delete -g $RG --nsg-name "$nsg" -n
  az network nsg rule create -g $RG --nsg-name "$nsg" -n ssh-from-builder --priority 1000 --access Allow \
    --protocol Tcp --direction Inbound --destination-port-ranges 22 --source-address-prefixes "$MYIP/32" -o none
}

case "${1:-status}" in
  quota)
    az vm list-usage --location "$REGION" -o table | grep -iE "NVADSA10|NCADSA100|NCADSH100|T4" ;;
  train)
    az group create -n $RG -l "$REGION" --tags $TAGS -o none
    az vm create -g $RG -n bree-train --image Ubuntu2204 --size Standard_NC24ads_A100_v4 \
      --admin-username azureuser --ssh-key-values ~/.ssh/id_rsa.pub --os-disk-size-gb 1024 \
      --storage-sku Premium_LRS --nsg-rule SSH --tags $TAGS -o none
    lock_ssh bree-train
    az vm auto-shutdown -g $RG -n bree-train --time "$SHUTDOWN_UTC" -o none
    az vm extension set -g $RG --vm-name bree-train --name NvidiaGpuDriverLinux --publisher Microsoft.HpcCompute -o none
    IP=$(az vm show -d -g $RG -n bree-train --query publicIps -o tsv)
    ssh -o StrictHostKeyChecking=accept-new azureuser@"$IP" \
      'nvidia-smi && sudo apt-get update -qq && sudo apt-get install -y -qq tmux git && \
       git clone https://github.com/kiromoussa/bree.git bree-vision && cd bree-vision && \
       git checkout claude/new-session-v6zi52 && ./scripts/setup.sh && make test'
    echo "bree-train at $IP" ;;
  sim)
    # NVIDIA Isaac Automator (https://github.com/isaac-sim/IsaacAutomator). Pin a stable Isaac Sim tag. It reads
    # NGC_API_KEY from the environment and prompts for anything else (flags checked against its deploy-azure, 2026-09-30).
    # Azure A10 VMs ship the GRID 570 driver: Isaac Sim 5.1 wants 580.65+, 4.5 wants 535.129+ -> default 4.5.0.
    : "${NGC_API_KEY:?export NGC_API_KEY first (ngc.nvidia.com -> Setup -> API key)}"
    ISAAC_TAG="${ISAAC_TAG:-4.5.0}"
    [ -d ../IsaacAutomator ] || git clone https://github.com/isaac-sim/IsaacAutomator ../IsaacAutomator
    (cd ../IsaacAutomator && ./build && ./deploy-azure bree-sim --instance-type Standard_NV36ads_A10_v5 \
      --region "$REGION" --isaacsim "$ISAAC_TAG" --resource-group $RG --tags project=bree \
      --ingress-cidrs "$MYIP/32")
    echo "Isaac Sim $ISAAC_TAG deployed; see ../IsaacAutomator/state/bree-sim. Log the tag in PROGRESS.md." ;;
  stop)
    az vm list -g $RG --query "[].name" -o tsv | xargs -r -n1 az vm deallocate -g $RG --no-wait -n
    az vm list -d -g $RG -o table ;;
  status)
    az vm list -d -g $RG -o table 2>/dev/null || echo "no $RG yet" ;;
esac
