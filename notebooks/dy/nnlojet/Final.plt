
set terminal pdfcairo enhanced color size 10cm,10cm dashed  # font "Iosevka,7" fontscale 0.65
set encoding utf8
set style fill transparent pattern border
set pointintervalbox 0.0001

set style line 1 dt 1   lc rgb '#B5B5AC' lw 1.2 pt 13 ps 0.3 # blue
set style line 2 dt 1   lc rgb '#009BBF' lw 1.2 pt 13 ps 0.3 # orange
set style line 3 dt 1   lc rgb '#FF9500' lw 1.2 pt 13 ps 0.3 # yellow
set style line 4 dt 1   lc rgb '#8F76D6' lw 1.2 pt 13 ps 0.3 # purple
set style line 9 dt 1   lc rgb '#D53E4F' lw 1.2 pt 7  ps 0.3 # red


if (!exists("PATH"))  PATH  = 'data'
if (!exists("OBS"))   OBS   = 'abs_yz'
if (!exists("REF"))   REF   = 'NLO'
if (!exists("PARTS")) PARTS = 'LO NLO NNLO'

den = PATH.'/'.REF.'.'.OBS.'.dat'

#> parse the `den` histogram file to exract some information on the data structure we're dealing with
nx = 0;
eval system("awk 'BEGIN{bin_acc=0.;bin_num=-1;bin_lst=0.;y_min=0.;y_max=0.;}$1!~/^#/{bin_num++;if(bin_num==0){x_min=$1;x_max=$3;};if(bin_num>0){bin_acc+=($3-$1)/bin_lst;bin_vals[bin_num]=($3-$1)/bin_lst;}bin_lst=$3-$1;if($1<x_min){x_min=$1};if($3>x_max){x_max=$3};if($4!=0.){if(y_min==0.||$4<y_min){y_min=$4;};if(y_max==0.||$4>y_max){y_max=$4;};};}$1~/^#labels:/{printf(\"ncol = %d;\", gensub(/.+\\[([0-9]+)\\]$/, \"\\\\1\", \"g\", $(NF)));}$1~/^#nx:/{printf(\"nx = %d;\", $2);}END{printf(\"bin_fac = %e;\",bin_acc/bin_num);printf(\"x_min = %e; x_max = %e;\",x_min,x_max);printf(\"y_min = %e; y_max = %e;\",y_min,y_max);}' ".den)
#> only plots for distributions
if (nx != 3) quit;
if (y_max <= y_min) quit;
#print(bin_fac)

#> in case we have a channel breakdown, we only care about the `tot_` columns!
eval system(sprintf("awk 'BEGIN{nx=%d}$1~/^#labels/{icol=nx+1;do{icol++;}while($(icol+1)~/^tot_/);printf(\"ncol_tot = %d;\", gensub(/.+\\[([0-9]+)\\]$/, \"\\\\1\", \"g\", $(icol)));}' ",nx).den)


#> absolute predictions:  x_low, x_ctr, x_upp, y_ctr, y_err, y_min, y_max
yvalue(part) = "< cat ".part." | ".sprintf("awk 'BEGIN{nx=%d;ncol=%d;ncol_tot=%d;}$1!~/^#/{for(i=1;i<=nx;i++){printf(\"%e \",$(i));};y_ctr=$(nx+1);y_err=$(nx+2);y_min=y_ctr;y_max=y_ctr;for(i=nx+1;i<ncol_tot;i+=2){if($(i)<y_min){y_min=$(i);};if($(i)>y_max){y_max=$(i);};};printf(\" %e %e %e %e \\n\",y_ctr*1e-3,y_min*1e-3,y_max*1e-3,y_err*1e-3);}'", nx,ncol,ncol_tot)
#> relative predictions:  x_low, x_ctr, x_upp, r_ctr, r_err, r_min, r_max
yratio(part) = "< paste ".part." ".den." | ".sprintf("awk 'BEGIN{nx=%d;ncol=%d;ncol_tot=%d;}$1!~/^#/&&$1==$(ncol+1)&&$(nx)==$(ncol+nx){for(i=1;i<=nx;i++){printf(\"%e \",$(i));};y_ref=$(ncol+nx+1);if(y_ref==0.){printf(\"\\n\");next;};y_ctr=$(nx+1);y_err=$(nx+2);y_min=y_ctr;y_max=y_ctr;for(i=nx+1;i<ncol_tot;i+=2){if($(i)<y_min){y_min=$(i);};if($(i)>y_max){y_max=$(i);};};printf(\" %e %e %e %e \\n\",y_ctr/y_ref,y_min/y_ref,y_max/y_ref,y_err/y_ref);}'", nx,ncol,ncol_tot)
#print(yvalue(PATH.'/'.'NLO'.'.'.OBS.'.dat'))


set output 'nnlojet_'.OBS.'.pdf'

set xrange[x_min-(x_max-x_min)*1e-6:x_max+(x_max-x_min)*1e-6]
if (bin_fac > 1.1) {
  set log x
  bin_loc(xlow,xupp,pos) = exp(log(xlow) + pos*(log(xupp)-log(xlow)))
} else {
  unset log x
  bin_loc(xlow,xupp,pos) = xlow + pos*(xupp-xlow)
}
set yrange [*:*]
if (y_max/y_min > 1e2) {
  set log y
  set format y '%g'
} else {
  unset log y
  set format y '%g'
}

set lmargin 10
set key top left horizontal

if (exists("SETTINGS")) eval(SETTINGS)

set multiplot layout 2,1 title '' noenhanced
set key top right outside
set ylabel 'dσ / d|Y_{ℓℓ}| [pb]'
set format x ''
unset xlabel
plot \
  for [i=1:words(PARTS)] yvalue(PATH.'/'.word(PARTS, i).'.'.OBS.'.dat')." | awk '$1!~/^#/{print $0;$1=$3;print $0;}'" u 1:5:6 w filledc ls i fs transparent solid 0.3 t word(PARTS, i) noenhanced,  \
  for [i=1:words(PARTS)] for [j=4:6] yvalue(PATH.'/'.word(PARTS, i).'.'.OBS.'.dat')." | awk '$1!~/^#/{print $0;$1=$3;print $0;}'" u 1:j w l ls i notitle, \
  "../ATLAS.dat" u ($1+0.5*($2-$1)):3:($3*$4*1e-2) w err ls 9 t "ATLAS"
set ylabel 'Ratio to '.REF noenhanced
unset log y
set format x '%g'
set xlabel "|Y_{ℓℓ}|"
plot \
  for [i=1:words(PARTS)] yratio(PATH.'/'.word(PARTS, i).'.'.OBS.'.dat')." | awk '$1!~/^#/{print $0;$1=$3;print $0;}'" u 1:5:6 w filledc ls i fs transparent solid 0.3 notitle,  \
  for [i=1:words(PARTS)] for [j=4:6] yratio(PATH.'/'.word(PARTS, i).'.'.OBS.'.dat')." | awk '$1!~/^#/{print $0;$1=$3;print $0;}'" u 1:j w l ls i notitle,  \
  "< paste ../ATLAS.dat ".PATH.'/'.REF.'.'.OBS.".dat | awk '$1<2.8{print}'" u ($1+0.5*($2-$1)):($3*1e3/$8):($3*$4*1e-2*1e3/$8) w err ls 9 notitle
  # for [i=1:words(PARTS)] yratio(PATH.'/'.word(PARTS, i).'.'.OBS.'.dat') u (bin_loc($1,$3,0.25+i*0.5/(words(PARTS)+1.))):4:7 w err ls i ps 0.2 notitle
unset multiplot

unset output

