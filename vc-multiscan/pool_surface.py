"""Stream a surface volume into a physical-scale-aware OME-Zarr representation.

Requires verified physical voxel size; relative pyramid factors are unsafe.
python pool_surface.py render.zarr fixed.zarr --voxel-um 4.681 --spacing-evidence meta.json
Uses factor 2 and records actual 9.362 um, not a guessed 9.6 um.
"""
import argparse,json
from pathlib import Path
import numpy as np
import zarr
from numcodecs import Blosc

def choose_factors(voxel_um,target_um,tolerance=.1):
    voxel=np.broadcast_to(np.asarray(voxel_um,dtype=float),(3,)).copy()
    if np.any(~np.isfinite(voxel)) or np.any(voxel<=0) or not np.isfinite(target_um) or target_um<=0:
        raise ValueError('Spacing must be finite and positive')
    factors=np.maximum(1,np.floor(target_um/voxel+.5).astype(int))
    actual=voxel*factors
    if np.any(np.abs(actual/target_um-1)>tolerance):
        raise ValueError(f'Integer pooling cannot reach requested spacing within tolerance: {actual.tolist()}')
    return tuple(map(int,factors)),tuple(map(float,actual))

def block_mean(a,factors):
    """Mean valid voxels in partial boundary blocks, without zero padding."""
    out=a.astype(np.float32)
    for axis,factor in enumerate(factors):
        starts=np.arange(0,out.shape[axis],factor)
        counts=np.minimum(factor,out.shape[axis]-starts)
        shape=[1]*out.ndim;shape[axis]=len(counts)
        out=np.add.reduceat(out,starts,axis=axis)/counts.reshape(shape)
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source');p.add_argument('output');p.add_argument('--dataset',default='0')
    p.add_argument('--voxel-um',type=float,nargs='+',required=True,help='One isotropic value or three verified values in Z,Y,X order')
    p.add_argument('--spacing-evidence',required=True,help='Path or URL supporting physical input spacing')
    p.add_argument('--target-um',type=float,default=9.6);p.add_argument('--tolerance',type=float,default=.1)
    p.add_argument('--xy-chunk',type=int,default=128)
    args=p.parse_args()
    if len(args.voxel_um) not in (1,3): p.error('--voxel-um needs one or three values')
    if args.xy_chunk<1 or args.tolerance<0: p.error('Invalid chunk size or tolerance')
    if Path(args.output).exists(): raise FileExistsError('Refusing to overwrite existing output')
    factors,actual=choose_factors(args.voxel_um,args.target_um,args.tolerance)
    source=zarr.open_group(args.source,mode='r');arr=source[args.dataset]
    if arr.ndim!=3: raise ValueError('Expected Z,Y,X surface volume')
    shape=tuple((n+f-1)//f for n,f in zip(arr.shape,factors))
    root=zarr.open_group(args.output,mode='w');root.attrs['repair_complete']=False
    dst=root.create_dataset('0',shape=shape,chunks=(shape[0],args.xy_chunk,args.xy_chunk),dtype=arr.dtype,
                            compressor=Blosc(cname='zstd',clevel=3,shuffle=Blosc.BITSHUFFLE))
    for y in range(0,shape[1],args.xy_chunk):
        for x in range(0,shape[2],args.xy_chunk):
            y1,x1=min(shape[1],y+args.xy_chunk),min(shape[2],x+args.xy_chunk)
            block=arr[:,y*factors[1]:y1*factors[1],x*factors[2]:x1*factors[2]]
            mean=block_mean(block,factors)
            if np.issubdtype(arr.dtype,np.integer): mean=np.rint(mean)
            dst[:,y:y1,x:x1]=mean.astype(arr.dtype)
    root.attrs.update(multiscales=[dict(version='0.4',axes=[dict(name=n,type='space',unit='micrometer') for n in 'zyx'],
                      datasets=[dict(path='0',coordinateTransformations=[dict(type='scale',scale=list(actual))])])],
                      repair_complete=True,source=args.source,source_dataset=args.dataset,
                      input_voxel_um=np.broadcast_to(np.asarray(args.voxel_um,float),(3,)).tolist(),
                      pooling_factors_zyx=list(factors),requested_voxel_um=args.target_um,
                      spacing_evidence=args.spacing_evidence,
                      edge_policy='mean of valid voxels; last block may have fewer voxels')
    print(json.dumps(dict(shape=shape,factors=factors,actual_voxel_um=actual)))

if __name__=='__main__':main()
