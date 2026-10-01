#!/usr/bin/env bash
# BREE Azure GPU VMs. Everything in resource group bree-rg, tagged project=bree, auto-shutdown on,
# SSH only and only from this machine's current public IP.
#   scripts/azure_gpu.sh quota  [region]   # GPU quota in a region
#   scripts/azure_gpu.sh train  [region]   # bree-train: 1x A100 80GB (NC24ads_A100_v4), 1 TB disk, driver, repo, make test
#   scripts/azure_gpu.sh sim    [region]   # bree-sim:   1x A10 24GB  (NV36ads_A10_v5), Isaac Sim 4.5.0 container via Isaac Automator v3.13.0
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
    # NVIDIA Isaac Automator v3.13.0, the last release that deploys the Isaac Sim *container*
    # (--isaac-image). v4.x builds Isaac Sim from github.com/isaac-sim/IsaacSim tags, which start at v5.0.0.
    # v3.13.0 installs the Azure GRID driver 535.161.08 (Isaac Sim 4.5 needs >= 535.129.03) and runs the
    # container with ~/results -> /results and ~/uploads -> /uploads (src/ansible/roles/isaac/templates/isaacsim.sh).
    # Its terraform opens SSH/VNC/NoMachine to 0.0.0.0/0, so lock_ssh below replaces those rules.
    # First run: `az login` inside the Automator container prints a device code; a human completes it once.
    # Never `./destroy bree-sim`: the RG is imported into its terraform state (the azurerm provider refuses
    # to delete a non-empty RG, but don't rely on that). Use `stop` below.
    : "${NGC_API_KEY:?export NGC_API_KEY first (ngc.nvidia.com -> Setup -> API key)}"
    ISAAC_IMAGE="${ISAAC_IMAGE:-nvcr.io/nvidia/isaac-sim:4.5.0}"
    IA=../IsaacAutomator
    [ -d $IA ] || git clone --depth 1 --branch v3.13.0 https://github.com/isaac-sim/IsaacAutomator $IA
    printf '#!/bin/sh\n# bree: do not start the Isaac Sim GUI container on boot\n' > $IA/uploads/autorun.sh
    az group create -n $RG -l "$REGION" --tags $TAGS -o none
    RG_ID=$(az group show -n $RG --query id -o tsv)
    (cd $IA && ./run ./deploy-azure --deployment-name bree-sim --region "$REGION" \
      --isaac-instance-type Standard_NV36ads_A10_v5 --isaac --isaac-image "$ISAAC_IMAGE" --ngc-api-key "$NGC_API_KEY" \
      --oige no --isaaclab no --in-china no --upload --resource-group "$RG_ID" --ingress-cidrs "$MYIP/32")
    VM=$(az vm list -g $RG --query "[?contains(name,'bree-sim')].name | [0]" -o tsv)
    lock_ssh bree-sim
    az resource list -g $RG --query "[?contains(name,'bree-sim')].id" -o tsv | \
      xargs -r -n1 az resource tag --is-incremental --tags $TAGS -o none --ids
    az vm auto-shutdown -g $RG -n "$VM" --time "$SHUTDOWN_UTC" -o none
    echo "Isaac Sim ($ISAAC_IMAGE) on $VM: ssh -i $IA/state/bree-sim/key.pem ubuntu@$(az vm show -d -g $RG -n "$VM" --query publicIps -o tsv)"
    echo "Log the image tag in PROGRESS.md." ;;
  stop)
    az vm list -g $RG --query "[].name" -o tsv | xargs -r -n1 az vm deallocate -g $RG --no-wait -n
    az vm list -d -g $RG -o table ;;
  status)
    az vm list -d -g $RG -o table 2>/dev/null || echo "no $RG yet" ;;
esac
