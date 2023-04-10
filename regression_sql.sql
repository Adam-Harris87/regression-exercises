show databases;
use zillow;
show tables;

-- bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, and fips 
-- for all 'Single Family Residential' properties

describe properties_2016; 
select * from properties_2016 order by id;
select * from properties_2017;
select * from propertylandusetype;
describe properties_2017;
select count(*) from properties_2017;
select * from typeconstructiontype;
select * from airconditioningtype;
select * from predictions_2017;

SELECT propertylandusetypeid
FROM propertylandusetype
WHERE propertylandusedesc IN ("Single Family Residential", "Inferred Single Family Residential")
;

SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
FROM properties_2016
WHERE propertylandusetypeid IN(
	SELECT propertylandusetypeid
	FROM propertylandusetype
	WHERE propertylandusedesc = "Single Family Residential")
;

WITH cte_sfr as(
	SELECT * 
    FROM properties_2017
    WHERE propertylandusetypeid IN (
		SELECT propertylandusetypeid
		FROM propertylandusetype
		WHERE propertylandusedesc IN( 
			"Single Family Residential", "Inferred Single Family Residential"))
	AND parcelid IN (
		SELECT parcelid
        FROM predictions_2017)
)

SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet,
	garagecarcnt, garagetotalsqft, lotsizesquarefeet, poolcnt, poolsizesum,
	yearbuilt, fips, regionidcity, taxvaluedollarcnt, taxamount
FROM cte_sfr
;

SELECT *
FROM properties_2017
WHERE propertylandusetypeid IN(
 	SELECT propertylandusetypeid
	FROM propertylandusetype
 	WHERE propertylandusedesc = "Single Family Residential")
LIMIT 5
;

select * from predictions_2017;