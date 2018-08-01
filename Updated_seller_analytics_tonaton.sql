DROP TABLE if exists tonaton_temp.agg_seller_analytics_shop_metrics_temp_ani;

CREATE TABLE tonaton_temp.agg_seller_analytics_shop_metrics_temp_ani distkey 
(
  account_id   
)
AS
(
 SELECT COALESCE(t1.account_id,t2.account_id) AS account_id,
       COALESCE(t1.shop_id,t2.shop_id) AS shop_id,
       COALESCE(t1.shop_slug,t2.shop_slug) AS shop_slug,
       COALESCE(t1.shop_name,t2.shop_name) AS shop_name,
       COALESCE(t1.email,t2.email) AS shop_email,
       COALESCE(t1.phone_numbers,t2.phone_numbers) AS shop_contact_number,
       COALESCE(t1.membership_name,t2.membership_name) AS membership_category,
       COALESCE(t1.membership_level,t2.membership_level) AS shop_member_type,
       SUM(nvl (t1.shop_views,0) + nvl (t2.shop_views,0)) AS shop_views,
       0 AS ad_pageviews,
       0 AS published_ads_count,
       0 AS leads,
       DATE_TRUNC('month',sysdate) AS date_value
FROM (SELECT subs.account_id,
             shops.shop_id,
             shops.shop_slug,
             MAX(shops.shop_name) AS shop_name,
             MAX(shops.email_address) AS email,
             MAX(shops.phone_numbers) AS phone_numbers,
             MAX(CASE WHEN mem.membership_name IS NULL THEN 'NA' ELSE mem.membership_name END) AS membership_name,
             MAX(CASE WHEN subs.membership_level = 'plus' THEN 'Business Plus' WHEN subs.membership_level = 'premium' THEN 'Business premium' ELSE 'NA' END) AS membership_level,
             COUNT(DISTINCT evn.event_id) AS shop_views
FROM tonaton_bidb.dim_subscriptions subs 
  LEFT join tonaton_bidb.dim_membership_master mem ON mem.membership_key = subs.membership_key
        LEFT JOIN tonaton_bidb.dim_shops shops ON subs.account_id = shops.account_id
        LEFT JOIN tonaton_insights.fact_snowplow_events evn
               ON ( (shops.shop_slug = REPLACE (REPLACE (REPLACE (REPLACE (evn.page_urlpath,'/en/shops/',''),'/bn/shops/',''),'/si/shops/',''),'/ta/shops/','')
               OR shops.shop_id = REPLACE (REPLACE (REPLACE (REPLACE (evn.page_urlpath,'/en/shops/',''),'/bn/shops/',''),'/si/shops/',''),'/ta/shops/',''))
              AND page_url ilike '%/shops/%'
              AND event = 'page_view'
              AND (evn.app_id = 'TONATON'
               OR evn.app_id = 'TONATON_NOJS')
              AND page_urlquery IS NULL)
        LEFT JOIN tonaton_bidb.dim_date dd ON dd.actual_date = TRUNC (evn.collector_tstamp)
      WHERE subs.expires_at >= (last_day(sysdate -INTERVAL '1 month') +INTERVAL '1 day')::datetime
      AND   dd.month_diff IN (1)
      GROUP BY 1,
               2,
               3) t1 
FULL JOIN (SELECT subs.account_id,
                    shops.shop_id,
                    shops.shop_slug,
                    MAX(shops.shop_name) AS shop_name,
                    MAX(shops.email_address) AS email,
                    MAX(shops.phone_numbers) AS phone_numbers,
                    MAX(CASE WHEN mem.membership_name IS NULL THEN 'NA' ELSE mem.membership_name END) AS membership_name,
                    MAX(CASE WHEN subs.membership_level = 'plus' THEN 'Business Plus' WHEN subs.membership_level = 'premium' THEN 'Business premium' ELSE 'NA' END) AS membership_level,
                    COUNT(DISTINCT evn.event_id) AS shop_views
             FROM tonaton_bidb.dim_subscriptions subs LEFT join tonaton_bidb.dim_membership_master mem ON mem.membership_key = subs.membership_key
               LEFT JOIN tonaton_bidb.dim_shops shops ON subs.account_id = shops.account_id
               LEFT JOIN tonaton_insights.fact_snowplow_events evn
                      ON (shops.shop_id = evn.se_property
                     AND se_label = 'Shop Id'
                     AND (se_action = 'ShopDetail'
                      OR se_action = 'ShopInformation'
                      OR se_action = 'Shop Detail')
                     AND (evn.app_id = 'TONATON_IOS'
                      OR evn.app_id = 'TONATON_AND'))
               LEFT JOIN tonaton_bidb.dim_date dd ON dd.actual_date = TRUNC (evn.collector_tstamp)
             WHERE subs.expires_at >= (last_day(sysdate -INTERVAL '1 month') +INTERVAL '1 day')::datetime
             AND   dd.month_diff IN (1)
             GROUP BY 1,
                      2,
                      3) t2 ON t1.account_id = t2.account_id
GROUP BY 1,
         2,
         3,
         4,
         5,
         6,
         7,
         8);

UPDATE tonaton_temp.agg_seller_analytics_shop_metrics_temp_ani
   SET published_ads_count = src.published_ads_count,
       ad_pageviews = src.ad_pageviews,
       leads = src.leads
FROM (SELECT account_id,
             COUNT(DISTINCT ad_id) AS published_ads_count,
             COUNT(DISTINCT CASE WHEN mashup.total_conversions > 0 THEN mashup.domain_userid END) AS converted_users,
             SUM(page_views) AS ad_pageviews,
             COUNT(DISTINCT CASE WHEN mashup.total_leads > 0 OR mashup.show_number_conversion > 0 OR mashup.send_message_leads > 0 OR mashup.place_order_leads > 0 THEN mashup.domain_userid END) +(SELECT COUNT(DISTINCT buyer_account_id)
                                                                                                                                                                                                    FROM tonaton_bidb.fact_conversations conv
                                                                                                                                                                                                    WHERE conv.seller_account_id = mashup.account_id
                                                                                                                                                                                                    AND   conv.conversation_timestamp >= (last_day(sysdate -INTERVAL '1 month') +INTERVAL '1 day')::datetime)
      AS leads
      FROM tonaton_insights.agg_ad_metrics_mashup mashup
        JOIN tonaton_bidb.dim_date dd ON TO_CHAR (event_date,'YYYYMMDD') = dd.full_date
      WHERE dd.month_diff IN (1)
      GROUP BY 1) src
WHERE src.account_id = agg_seller_analytics_shop_metrics_temp_ani.account_id;

