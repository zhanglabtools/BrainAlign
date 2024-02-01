path.bin <- 'D:/Research_programs/single_cell/molecular-atlas-master/bin'
path.matrices <-'D:/Research_programs/single_cell/figures'

load(paste(path.matrices ,'atlasspots.RData',sep='/'))
load(paste(path.matrices , 'vivid-colors.RData', sep='/'))

#Only loading if not existing, takes some time
if(!exists('aligned.atlas'))
  load(paste(path.matrices ,'alignedAtlas.RData',sep='/'))

if(!exists('atlas.stereo'))
  load(paste(path.matrices ,'atlasStereo.RData',sep='/'))

source(paste(path.bin,'plotFunctions.R',sep='/'))
source(paste(path.bin,'execFunctions.R',sep='/'))
source(paste(path.bin,'araAtlasFunctions.R',sep='/'))
source(paste(path.bin,'smoothingFunctions.R',sep='/'))
source(paste(path.bin,'layerFunctions.R',sep='/'))
source(paste(path.bin,'icFunctions.R',sep='/'))
source(paste(path.bin,'tsneFunctions.R',sep='/'))
source(paste(path.bin,'similarityIndexFunctions.R',sep='/'))
source(paste(path.bin,'scRemapFunctions.R',sep='/'))
source(paste(path.bin,'constantParameters.R',sep='/'))
source(paste(path.bin,'allenAnnotationsFunctions.R',sep='/'))
source(paste(path.bin,'meshCutsFunctions.R',sep='/'))